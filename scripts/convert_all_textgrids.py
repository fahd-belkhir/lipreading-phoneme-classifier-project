import os
from textgrid import TextGrid

input_dir = r"E:\validation_output"
output_dir = r"E:\validation_output"  # Can be different if you want

def convert_textgrid_to_phn(textgrid_path, output_path):
    tg = TextGrid.fromFile(textgrid_path)
    phones = tg.getFirst('phones')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for interval in phones:
            phoneme = interval.mark.strip()
            start = round(interval.minTime, 3)
            end = round(interval.maxTime, 3)
            if phoneme:
                f.write(f"{phoneme} {start} {end}\n")

# Loop through all .TextGrid files
for file in os.listdir(input_dir):
    if file.endswith(".TextGrid"):
        base = os.path.splitext(file)[0]
        tg_path = os.path.join(input_dir, file)
        phn_path = os.path.join(output_dir, f"{base}.phn")
        
        print(f"Converting {file} → {base}.phn")
        convert_textgrid_to_phn(tg_path, phn_path)

print("✅ All .TextGrid files converted to .phn format.")
