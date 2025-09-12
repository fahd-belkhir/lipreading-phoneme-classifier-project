import os
import cv2

# === Set your test paths ===
frames_folder = r"E:\lrs2_mouth_crops\5535415699068794046_00001"
phn_file = r"E:\mfa_output\5535415699068794046_00001.phn"
fps = 25

# Load phoneme timings
phonemes = []
with open(phn_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            label, start, end = parts
            phonemes.append((label, float(start), float(end)))

# Align each frame
aligned = []
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
for i, file in enumerate(frame_files):
    timestamp = i / fps
    matching_phn = next((label for label, start, end in phonemes if start <= timestamp < end), "SIL")
    aligned.append((file, matching_phn))

# Print aligned pairs
for file, label in aligned:
    print(f"{file} → {label}")
