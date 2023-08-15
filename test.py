import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

# print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device)
model.eval()
image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)

# text = clip.tokenize(["a tiger", "a cat", "a dog", "a bear"]).to(device)
text = clip.tokenize(["老虎", "猫", "狗", "熊"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

# [[2.5376787e-03 9.9683857e-01 4.3544930e-04 1.8830669e-04]] for ["a tiger", "a cat", "a dog", "a bear"]
# [[1.9532440e-03 9.9525285e-01 2.2442457e-03 5.4962368e-04]] for ["老虎", "猫", "狗", "熊"]
