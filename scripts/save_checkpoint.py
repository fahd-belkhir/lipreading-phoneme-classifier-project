import torch
from model import LipReadingModel
import json

# === Load phoneme vocab ===
with open("phoneme_vocab.json", "r", encoding="utf-8") as f:
    phoneme2idx = json.load(f)

# === Create model instance ===
model = LipReadingModel(num_classes=len(phoneme2idx))

# === Load weights from memory (if you're still in Python) or from last saved (optional) ===
# Replace this line with your own model's current in-memory state_dict if you still have it
# If you don't, just skip loading and save a fresh model
# model.load_state_dict(torch.load("checkpoints/model_epoch_1.pt"))

# === Save checkpoint ===
torch.save(model.state_dict(), "checkpoints/model_resume.pt")
print("✅ Model manually saved to checkpoints/model_resume.pt")
