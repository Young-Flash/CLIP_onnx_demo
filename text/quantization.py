import os
from pathlib import Path

import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process


def text_quantize():
    model = "clip-cn-text-encoder.onnx"
    model_prep = "clip-cn-text-encoder-quant-pre.onnx"
    model_quant = "clip-cn-text-encoder-quant-int8.onnx"

    cur_path = Path(os.curdir)

    quant_pre_process(model, model_prep)  # preprocess for quantization
    quantize_dynamic(
        cur_path / model_prep, cur_path / model_quant, weight_type=QuantType.QInt8
    )
