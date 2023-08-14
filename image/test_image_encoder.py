import onnxruntime as ort
from PIL import Image
import numpy as np
import clip


def test_encode_image(model_path: str, image_path: str, preprocess) -> np.ndarray:
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
    outputs = ort_session.run(None, {input_name: image_input.numpy()})
    return outputs[0]
