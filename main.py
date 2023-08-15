from image.image_encoder import get_image_encoder_onnx
from image.quantization import image_quantize
from image.test_image_encoder import test_encode_image

from text.text_encoder_onnx_cn import get_text_encoder_onnx_cn
from text.quantization import text_quantize
from text.test_text_encoder import test_encode_text

import numpy as np
import cn_clip.clip as clip
import torch

from torch import nn

device = "cpu"
model, preprocess = clip.load_from_name("ViT-B-16", device=device)

# get_image_encoder_onnx(model, preprocess)
# image_quantize()
# res_image = test_encode_image("clip-cn-image-encoder.onnx", "image.jpg", preprocess)
res_image = test_encode_image(
    "clip-cn-image-encoder-quant-int8.onnx", "image.jpg", preprocess
)

# get_text_encoder_onnx_cn(model)
# text_quantize()
# res_text = test_encode_text(
#     "clip-cn-text-encoder.onnx",
#     ["a tiger", "a cat", "a dog", "a bear"]
#     # ["老虎", "猫", "狗", "熊"]
# )
res_text = test_encode_text(
    "clip-cn-text-encoder-quant-int8.onnx",
    # ["a tiger", "a cat", "a dog", "a bear"]
    ["老虎", "猫", "狗", "熊"],
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

# Label probs: [[1.9535627e-03 9.9525201e-01 2.2446462e-03 5.4973643e-04]] for ["老虎", "猫", "狗", "熊"] (with clip-cn-image-encoder.onnx & clip-cn-text-encoder.onnx)
# Label probs: [[2.5380836e-03 9.9683797e-01 4.3553708e-04 1.8835040e-04]] for ["a tiger", "a cat", "a dog", "a bear"] (with clip-cn-image-encoder.onnx & clip-cn-text-encoder.onnx)

# Label probs: [[0.00884504 0.98652565 0.00179121 0.00283814]] for ["老虎", "猫", "狗", "熊"] (with clip-cn-image-encoder-quant-int8.onnx & clip-cn-text-encoder-quant-int8.onnx.onnx)
# Label probs: [[0.02240802 0.97132427 0.00435637 0.00191139]] for ["a tiger", "a cat", "a dog", "a bear"] (with clip-cn-image-encoder-quant-int8.onnx & clip-cn-text-encoder-quant-int8.onnx.onnx)
