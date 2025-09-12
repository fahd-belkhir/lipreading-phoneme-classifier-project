from textgrid import TextGrid
import os

def extract_phonemes(textgrid_path, output_path=None):
    tg = TextGrid.fromFile(textgrid_path)
    phones = tg.getFirst('phones')  # get phoneme tier

    results = []
    for interval in phones:
        phoneme = interval.mark.strip()
        start = round(interval.minTime, 3)
        end = round(interval.maxTime, 3)
        if phoneme:  # skip blanks
            results.append((phoneme, start, end))

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for phoneme, start, end in results:
                f.write(f"{phoneme} {start} {end}\n")
        print(f"Saved phonemes to {output_path}")
    else:
        for phoneme, start, end in results:
            print(f"{phoneme} {start} {end}")

    return results


# Example usage
if __name__ == "__main__":
    textgrid_file = "E:/mfa_output/00001.TextGrid"  # ← Replace with your actual file path
    output_file = "E:/mfa_output/00001.phn"         # ← Output as text file (optional)
    extract_phonemes(textgrid_file, output_file)
