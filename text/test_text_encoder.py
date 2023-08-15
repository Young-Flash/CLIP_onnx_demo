import onnxruntime as ort
from PIL import Image
import numpy as np
import cn_clip.clip as clip


def test_encode_text(model_quant: str, text: list) -> np.ndarray:
    ort_session = ort.InferenceSession(model_quant)
    input_name = ort_session.get_inputs()[0].name

    outputs = []
    for item in text:
        token_input: Tensor = clip.tokenize(item).to("cpu")
        output = ort_session.run(None, {input_name: token_input.numpy()})
        outputs.append(output[0])

    return outputs
