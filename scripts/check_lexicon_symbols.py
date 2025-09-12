import re
from pathlib import Path

phones_path = Path(r"C:\Users\belkh\Documents\MFA\mfa_input_val\dictionary\phones\phones.txt")
words_path = Path(r"C:\Users\belkh\Documents\MFA\mfa_input_val\dictionary\1_english_uk_mfa\words.txt")
lex_path = Path(r"C:\Users\belkh\Documents\MFA\mfa_input_val\dictionary\1_english_uk_mfa\lexicon.text_fst")

def load_symbols(p):
    syms = set()
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Symbol tables are usually "SYMBOL\tID"
            parts = re.split(r'\s+', line)
            if len(parts) >= 1:
                syms.add(parts[0])
    return syms

phones = load_symbols(phones_path)
words = load_symbols(words_path)

def is_float(s):
    try:
        float(s); return True
    except: return False

def is_int(s):
    try:
        int(s); return True
    except: return False

bad_count = 0
with open(lex_path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: 
            continue
        # Expected form (text FST arc):
        # src dst ilabel olabel [weight]
        parts = re.split(r'\s+', line)
        if len(parts) < 4:
            print(f"[Line {i}] Too few fields: {line}")
            bad_count += 1
            break

        src, dst, ilab, olab = parts[0], parts[1], parts[2], parts[3]
        # ignore numeric state IDs
        # ilab is a PHONE symbol, olab is a WORD symbol (or <eps>)
        if not (is_int(src) and is_int(dst)):
            print(f"[Line {i}] Non-integer state IDs: {parts[:2]} | {line}")
            bad_count += 1
            break

        # ilabel check
        if ilab != "<eps>" and ilab not in phones:
            print(f"[Line {i}] Unknown PHONE symbol in ilabel: '{ilab}' | line: {line}")
            bad_count += 1
            break

        # olabel check
        if olab != "<eps>" and olab not in words:
            print(f"[Line {i}] Unknown WORD symbol in olabel: '{olab}' | line: {line}")
            bad_count += 1
            break

        # optional weight in parts[4] is okay (float)
        if len(parts) >= 5 and not is_float(parts[4]):
            print(f"[Line {i}] Non-numeric weight: '{parts[4]}' | line: {line}")
            bad_count += 1
            break

if bad_count == 0:
    print("✅ No obvious symbol mismatches found in the first pass.")
