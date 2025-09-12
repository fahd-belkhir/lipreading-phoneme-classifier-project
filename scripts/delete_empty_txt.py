import os

input_root = r"E:\mfa_input_val"
deleted = []

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    os.remove(txt_path)
                    deleted.append(txt_path)
            except Exception as e:
                print(f"❌ Could not read {txt_path}: {e}")

print(f"🗑️ Deleted {len(deleted)} empty .txt files")
