import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# STEP 1: Run BLIP in a separate process
# --------------------------------------------------

caption_code = r'''
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent

print("Loading BLIP...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    dtype=torch.float16
).to("cuda")

image = Image.open(BASE_DIR / "input.png").convert("RGB")

print("Generating description...")

inputs = processor(
    images=image,
    return_tensors="pt"
).to("cuda", torch.float16)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

print("BLIP description:")
print(caption)

(BASE_DIR / "prompt.txt").write_text(
    caption,
    encoding="utf-8"
)

print("Saved prompt.txt")
'''

caption_file = BASE_DIR / "_caption_temp.py"

caption_file.write_text(caption_code, encoding="utf-8")

print("=== CAPTIONING IMAGE ===")

subprocess.run(
    [sys.executable, str(caption_file)],
    cwd=BASE_DIR,
    check=True
)

# Delete the temporary caption script
caption_file.unlink(missing_ok=True)

print("\nBLIP process completely finished.")
print("RAM/VRAM should now be released.\n")


# --------------------------------------------------
# STEP 2: Run the old image-to-image AI
# --------------------------------------------------

print("=== RUNNING OLD AI ===")

subprocess.run(
    [sys.executable, str(BASE_DIR / "old_ai_image.py")],
    cwd=BASE_DIR,
    check=True
)

print("\n=== DONE ===")
print("Created output.png")