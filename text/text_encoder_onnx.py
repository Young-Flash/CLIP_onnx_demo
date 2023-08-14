import clip
import torch
from text.text_encoder import TextEncoder


def get_text_encoder_onnx(model):
    model.eval()

    text_encoder = TextEncoder(
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )

    missing_keys, unexpected_keys = text_encoder.load_state_dict(
        model.state_dict(), strict=False
    )

    text_encoder.eval()

    input_tensor = clip.tokenize("a cat").to("cpu")
    traced_model = torch.jit.trace(text_encoder, input_tensor)

    onnx_filename = "clip-text-encoder.onnx"

    torch.onnx.export(text_encoder, input_tensor, onnx_filename)
