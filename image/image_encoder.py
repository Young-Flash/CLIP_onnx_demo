import clip
from PIL import Image
from torch import Tensor
import torch


def get_image_encoder_onnx(model, preprocess):
    i = Image.open("image.jpg")
    input_tensor: Tensor = preprocess(i).unsqueeze(0).to("cpu")
    vit = model.visual
    vit.eval()

    onnx_filename = "clip-cn-image-encoder.onnx"
    torch.onnx.export(vit, input_tensor, onnx_filename)
