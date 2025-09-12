import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CTCLoss
from model_vit import LipReadingViTModel as LipReadingModel  # ✅ New
from dataset_loader import CachedTensorDataset
from train_utils import collate_fn
import json
import os

# === Config ===
vocab_path = "phoneme_vocab.json"
tensor_cache_dir = r"E:\lrs2_tensor_cache"
batch_size = 4
num_epochs = 5
learning_rate = 1e-4
save_dir = "checkpoints"
resume_path = os.path.join(save_dir, "model_resume.pt")

os.makedirs(save_dir, exist_ok=True)

# === Dataset Wrapping ===
class SafeDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        try:
            return self.base_dataset[idx]
        except Exception:
            return None

def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return collate_fn(batch)

# === Main Training ===
if __name__ == "__main__":
    # === Load vocab ===
    with open(vocab_path, 'r', encoding='utf-8') as f:
        phoneme2idx = json.load(f)

    # === Dataset + DataLoader ===
    raw_dataset = CachedTensorDataset(tensor_cache_dir)
    dataset = SafeDataset(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=safe_collate_fn,
        num_workers=0,         # Important for Windows stability
        pin_memory=True
    )

    # === Model, Loss, Optimizer ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    model = LipReadingModel(num_classes=len(phoneme2idx)).to(device)
    criterion = CTCLoss(blank=phoneme2idx["<blank>"], zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Resume from checkpoint if available ===
    start_epoch = 0
    start_batch = 0
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch_idx"] + 1
        print(f"🔁 Resumed training from epoch {start_epoch}, batch {start_batch}")
    else:
        print("🚫 No checkpoint found. Starting from scratch.")

    # === Training Loop ===
    model.train()
    for epoch in range(start_epoch, num_epochs):
        print(f"\n🌀 Epoch {epoch + 1}")
        total_loss = 0
        valid_batches = 0

        for i, batch in enumerate(dataloader):
            if epoch == start_epoch and i < start_batch:
                continue  # Resume logic

            try:
                if batch is None:
                    continue

                inputs, targets, input_lengths, target_lengths = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)  # (T, B, C)
                log_probs = outputs.log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                valid_batches += 1

                if (i + 1) % 10 == 0:
                    print(f"[Batch {i+1}] Loss: {loss.item():.4f}")

                if (i + 1) % 500 == 0:
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, resume_path)
                    print(f"💾 Checkpoint saved at batch {i+1}")

            except Exception as e:
                print(f"⚠️ Batch {i} skipped due to error: {e}")
                continue

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"\n✅ Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        else:
            print(f"\n⚠️ Epoch {epoch+1} had no valid batches.")

        # Save at epoch end
        epoch_ckpt = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_ckpt)
        torch.save({
            'epoch': epoch + 1,
            'batch_idx': -1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, resume_path)
        print(f"💾 Saved model for epoch {epoch+1} → {epoch_ckpt}")
