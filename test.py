import torch
import clip
from PIL import Image
import numpy as np
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a tiger", "a cat", "a dog", "a bear"]).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

# Label probs: [[6.1091479e-02 9.3267566e-01 5.3717378e-03 8.6108845e-04]]
