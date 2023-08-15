import cn_clip.clip as clip
import torch
from text.text_encoder_cn import TextEncoderCN


def get_text_encoder_onnx_cn(model):
    model.eval()

    text_encoder = TextEncoderCN(
        embed_dim=512,
        vocab_size=21128,
        text_attention_probs_dropout_prob=0.1,
        text_hidden_act="gelu",
        text_hidden_dropout_prob=0.1,
        text_hidden_size=768,
        text_initializer_range=0.02,
        text_intermediate_size=3072,
        text_max_position_embeddings=512,
        text_num_attention_heads=12,
        text_num_hidden_layers=12,
        text_type_vocab_size=2,
    )

    missing_keys, unexpected_keys = text_encoder.load_state_dict(
        model.state_dict(), strict=False
    )

    text_encoder.eval()

    input_tensor = clip.tokenize("a cat").to("cpu")
    traced_model = torch.jit.trace(text_encoder, input_tensor)

    onnx_filename = "clip-cn-text-encoder.onnx"

    torch.onnx.export(text_encoder, input_tensor, onnx_filename)
