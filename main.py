from image.image_encoder import get_image_encoder_onnx
from image.quantization import image_quantize
from image.test_image_encoder import test_encode_image

from text.text_encoder_onnx import get_text_encoder_onnx
from text.quantization import text_quantize
from text.test_text_encoder import test_encode_text

import numpy as np
import clip
import torch

from torch import nn

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# get_image_encoder_onnx(model, preprocess)
# image_quantize()
# res_image = test_encode_image("clip-image-encoder.onnx", "image.jpg", preprocess)
res_image = test_encode_image(
    "clip-image-encoder-quant-int8.onnx", "image.jpg", preprocess
)

# get_text_encoder_onnx(model)
# text_quantize()
# res_text = test_encode_text(
#     "clip-text-encoder.onnx", ["a tiger", "a cat", "a dog", "a bear"]
# )
res_text = test_encode_text(
    "clip-text-encoder-quant-int8.onnx", ["a tiger", "a cat", "a dog", "a bear"]
)

# convert res_image, res_text to toech.tensor
res_image = torch.from_numpy(res_image)
res_text = torch.from_numpy(np.concatenate(res_text, axis=0))

# calculate logits, copy from [clip/model.py](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L358-L372)

res_image = res_image / res_image.norm(dim=1, keepdim=True)
res_text = res_text / res_text.norm(dim=1, keepdim=True)

# cosine similarity as logits
logit_scale = 100  # get from model.state_dict()
logits_per_image = logit_scale * res_image @ res_text.t()
logits_per_text = logits_per_image.t()

probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)

# Label probs: [[6.1091259e-02 9.3267584e-01 5.3716768e-03 8.6109847e-04]] (with clip-image-encoder.onnx & clip-text-encoder.onnx)
# Label probs: [[0.04703762 0.9391219  0.00990335 0.00393698]] (with clip-image-encoder-quant-int8.onnx & clip-text-encoder-quant-int8.onnx)
