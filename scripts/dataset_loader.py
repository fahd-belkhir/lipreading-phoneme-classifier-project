import os
import json
import torch
from torch.utils.data import Dataset
import cv2

class LipReadingDataset(Dataset):
    def __init__(self, json_dir, frames_root, vocab_path, transform=None):
        self.json_dir = json_dir
        self.frames_root = frames_root
        self.transform = transform

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.phoneme2idx = json.load(f)

        all_ids = [f.replace(".json", "") for f in os.listdir(json_dir) if f.endswith(".json")]
        self.sample_ids = []

        print("🔍 Filtering usable samples...")

        for sid in all_ids:
            frame_dir = os.path.join(frames_root, sid)
            json_path = os.path.join(self.json_dir, f"{sid}.json")

            if not os.path.isdir(frame_dir) or not os.path.exists(json_path):
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                frames = data.get("frames", [])
                phonemes = data.get("phonemes", [])

                if not isinstance(frames, list) or not isinstance(phonemes, list):
                    continue

                if len(frames) == 0 or len(phonemes) == 0:
                    continue

                if not any(os.path.exists(os.path.join(frame_dir, fn)) for fn in frames):
                    continue

                self.sample_ids.append(sid)

            except Exception as e:
                print(f"⚠️ Error processing {sid}: {e}")
                continue

        print(f"✅ {len(self.sample_ids)} valid samples loaded.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        json_path = os.path.join(self.json_dir, f"{sample_id}.json")
        frame_dir = os.path.join(self.frames_root, sample_id)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frame_files = data["frames"]
        phoneme_labels = data["phonemes"]

        # Clip to match length
        min_len = min(len(frame_files), len(phoneme_labels))
        frame_files = frame_files[:min_len]
        phoneme_labels = phoneme_labels[:min_len]

        # Load image frames
        frames = []
        for fname in frame_files:
            path = os.path.join(frame_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
            frames.append(img)

        if len(frames) == 0:
            raise IndexError(f"No valid frames in {sample_id}")

        # Map phonemes to indices
        phoneme_indices = [self.phoneme2idx.get(p, self.phoneme2idx["<blank>"]) for p in phoneme_labels[:len(frames)]]

        # Silence filtering
        sil_idx = self.phoneme2idx.get("SIL")
        if sil_idx is not None and len(phoneme_indices) > 0:
            sil_ratio = phoneme_indices.count(sil_idx) / len(phoneme_indices)
            if sil_ratio > 0.9:
                raise IndexError(f"Too much silence in {sample_id}")

        # CTC constraint
        max_len = (len(frames) - 1) // 2
        phoneme_indices = phoneme_indices[:max_len]

        if len(phoneme_indices) == 0:
            raise IndexError(f"CTC-trimmed sequence too short in {sample_id}")

        frames_tensor = torch.stack(frames)  # (T, C, H, W)
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)  # (L,)

        return frames_tensor, phoneme_tensor


class CachedTensorDataset(Dataset):
    def __init__(self, cache_dir):
        self.paths = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        return (
            data['frames'],        # (T, C, H, W)
            data['phonemes'],      # (L,)
            data['input_len'],     # tensor([T])
            data['target_len']     # tensor([L])
        )
