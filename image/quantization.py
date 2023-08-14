import os
from pathlib import Path

import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)


def image_quantize():
    model_fp32 = "clip-image-encoder.onnx"
    model_prep = "clip-image-encoder-quant-pre.onnx"
    model_quant = "clip-image-encoder-quant-int8.onnx"

    cur_path = Path(os.curdir)

    # preprocess for quantization
    quant_pre_process(model_fp32, model_prep)

    # nodes_to_exclude=['/conv1/Conv']) is necessary, to leave Conv layer as it is since ConvInteger(10) is NOT implemented in onnxruntime with data type int8.
    quantize_dynamic(
        cur_path / model_prep,
        cur_path / model_quant,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=["/conv1/Conv"],
    )
