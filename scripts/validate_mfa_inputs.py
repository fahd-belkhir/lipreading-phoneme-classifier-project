import os
import re

input_root = r"E:\mfa_input_val"
bad = []

pattern = re.compile(r"^[a-z\s']+$")  # allow lowercase words + apostrophes

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)
            wav_path = txt_path.replace(".txt", ".wav")
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    line = f.read().strip().lower()
                    if not line:
                        bad.append((txt_path, "Empty"))
                    elif not pattern.fullmatch(line.replace(" ", "")):
                        bad.append((txt_path, f"Weird characters: {line}"))
                    elif not os.path.exists(wav_path):
                        bad.append((txt_path, "Missing .wav"))
            except Exception as e:
                bad.append((txt_path, f"Unreadable: {e}"))

# Show results
print(f"\n🔍 {len(bad)} problematic files found:")
for path, reason in bad:
    print(f"⚠️ {path} → {reason}")
