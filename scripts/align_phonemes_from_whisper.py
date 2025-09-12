import os
import json
from g2p_en import G2p
from tqdm import tqdm

# === CONFIG ===
transcript_path = r"C:\Users\belkh\Desktop\Stage 4A\lipreading-phoneme-classifier\scripts\whisper_transcripts_val.json"  # Whisper output
output_dir = r"E:\mfa_phoneme_jsons_val"  # Aligned phoneme output
os.makedirs(output_dir, exist_ok=True)

g2p = G2p(pos=False)  # Avoid POS tagging

# === Load Whisper transcript JSON ===
with open(transcript_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Process each file ===
for sample_id, content in tqdm(data.items(), desc="🔄 Aligning phonemes"):
    out_path = os.path.join(output_dir, f"{sample_id}.json")
    if os.path.exists(out_path):
        continue

    segments = content.get("segments", [])
    phoneme_entries = []

    for seg in segments:
        for word in seg.get("words", []):
            word_text = word.get("word", "").strip()
            start = word.get("start")
            end = word.get("end")

            if not word_text or start is None or end is None:
                continue

            try:
                phonemes = [p for p in g2p(word_text) if p.strip()]
                if not phonemes:
                    continue
                duration = end - start
                step = duration / len(phonemes)
                for i, phn in enumerate(phonemes):
                    phoneme_entries.append({
                        "phoneme": phn,
                        "start": round(start + i * step, 4),
                        "end": round(start + (i + 1) * step, 4)
                    })
            except Exception as e:
                print(f"⚠️ G2P error on word '{word_text}' in {sample_id}: {e}")

    output = {
        "frames": [],
        "phonemes": phoneme_entries
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

print("✅ Phoneme alignment complete.")
