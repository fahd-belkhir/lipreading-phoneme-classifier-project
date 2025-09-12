import os

# === Config ===
input_root = r"E:\mfa_input_val"  # ← Change if needed
dict_path = r"C:\Users\belkh\Documents\MFA\pretrained_models\dictionary\english_uk_mfa.dict"  # ← Adjust if needed
log_path = "cleaned_files_log.txt"

# === Load known words from MFA dictionary ===
with open(dict_path, "r", encoding="utf-8") as f:
    known_words = {line.strip().split()[0].lower() for line in f if line.strip()}

print(f"✅ Loaded {len(known_words)} words from dictionary")

# === Process .txt files ===
cleaned = []
skipped = []
total = 0

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".txt"):
            total += 1
            txt_path = os.path.join(root, file)

            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    original_line = f.read().strip().lower()
                    words = original_line.split()
            except Exception as e:
                print(f"❌ Error reading {txt_path}: {e}")
                skipped.append(txt_path)
                continue

            # Filter unknown words
            filtered = [w for w in words if w in known_words]

            if filtered != words:
                if not filtered:
                    print(f"🚫 {txt_path} → All words invalid, skipping file")
                    skipped.append(txt_path)
                    continue

                # Save cleaned version
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(filtered) + "\n")

                print(f"🧼 Cleaned {txt_path} → Removed unknown words")
                cleaned.append(txt_path)

# === Summary ===
print(f"\n📊 Total .txt files scanned: {total}")
print(f"✅ Cleaned: {len(cleaned)}")
print(f"⚠️ Skipped (all words unknown or unreadable): {len(skipped)}")

# === Log cleaned files ===
with open(log_path, "w", encoding="utf-8") as log:
    for path in cleaned:
        log.write(path + "\n")
print(f"📝 Log saved to {log_path}")
