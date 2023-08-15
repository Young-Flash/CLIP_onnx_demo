import torch.nn as nn
from collections import OrderedDict
from .modeling_bert import BertConfig, BertModel

import torch
import numpy as np
import clip
import os
import collections
import six


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (
            self.input_resolution // patch_size,
            self.input_resolution // patch_size,
        )
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(
            N, L - 1, device=x.device, dtype=int
        )
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def default_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file=default_vocab(), do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def convert_tokens_to_string(tokens, clean_up_tokenization_spaces=True):
        """Converts a sequence of tokens (string) in a single string."""

        def clean_up_tokenization(out_string):
            """Clean up a list of simple English tokenization artifacts
            like spaces before punctuations and abreviated forms.
            """
            out_string = (
                out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
            )
            return out_string

        text = " ".join(tokens).replace(" ##", "").strip()
        if clean_up_tokenization_spaces:
            clean_text = clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def vocab_size(self):
        return len(self.vocab)


from .bert_tokenizer import FullTokenizer

_tokenizer = FullTokenizer()


class TextEncoderCN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # text
        vocab_size: int,
        text_attention_probs_dropout_prob: float,
        text_hidden_act: str,
        text_hidden_dropout_prob: float,
        text_hidden_size: int,
        text_initializer_range: float,
        text_intermediate_size: int,
        text_max_position_embeddings: int,
        text_num_attention_heads: int,
        text_num_hidden_layers: int,
        text_type_vocab_size: int,
        tokenizer=_tokenizer,
    ):
        super().__init__()

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.bert_config.hidden_size**-0.5
            )

    def forward(self, text):
        pad_index = self.tokenizer.vocab["[PAD]"]
        attn_mask = text.ne(pad_index).type(torch.float32)
        x = self.bert(text, attention_mask=attn_mask)[0].type(
            torch.float32
        )  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection
