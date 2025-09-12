import os
import cv2
import torch
import json
import numpy as np
from model_vit import LipReadingViTModel  # or model import LipReadingModel for GRU
from torchvision import transforms
from torch.nn.functional import log_softmax

# === Config ===
model_path = "checkpoints/model_epoch_5.pt"
vocab_path = "phoneme_vocab.json"
frame_folder = r"E:\cropped_validation\6233510664059729149_00001"  # 🔁 Change this to your sample path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class = LipReadingViTModel  # 🔁 Change to LipReadingModel if using GRU

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
idx2phoneme = {v: k for k, v in phoneme2idx.items()}

# === Load model ===
model = model_class(num_classes=len(phoneme2idx)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load frames ===
transform = transforms.Compose([
    transforms.ToTensor(),  # HWC → CHW, scales to [0, 1]
])

frame_files = sorted(os.listdir(frame_folder))
frames = []
for file in frame_files:
    if file.endswith(".jpg"):
        img = cv2.imread(os.path.join(frame_folder, file))
        img = cv2.resize(img, (112, 112))
        img = transform(img)
        frames.append(img)

if not frames:
    print("❌ No frames found in folder.")
    exit()

# === Prepare input tensor ===
frames_tensor = torch.stack(frames)  # (T, C, H, W)
frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

# === Run inference ===
with torch.no_grad():
    outputs = model(frames_tensor)  # (T, 1, num_classes)
    log_probs = log_softmax(outputs, dim=2)
    pred_indices = torch.argmax(log_probs, dim=2)[:, 0].tolist()

# === Decode CTC
decoded = []
prev = -1
for idx in pred_indices:
    if idx != prev and idx != phoneme2idx["<blank>"]:
        decoded.append(idx)
    prev = idx

predicted_phonemes = [idx2phoneme[i] for i in decoded]
print("🗣️ Predicted phoneme sequence:", predicted_phonemes)
