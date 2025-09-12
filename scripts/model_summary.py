import torch
import json
from model import LipReadingModel as GRUModel
from model_vit import LipReadingViTModel as ViTModel

# === Config ===
vocab_path = "phoneme_vocab.json"

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
num_classes = len(phoneme2idx)

# === Parameter Counter ===
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 {model.__class__.__name__} has {total:,} parameters ({total / 1e6:.2f}M)")
    return total

# === Instantiate both models ===
print("🔍 Instantiating models...\n")

gru_model = GRUModel(num_classes=num_classes)
vit_model = ViTModel(num_classes=num_classes)

# === Print parameter counts ===
print("📊 Parameter Summary:\n")

print("GRU-Based Model:")
count_parameters(gru_model)

print("\nViT-Based Model:")
count_parameters(vit_model)
