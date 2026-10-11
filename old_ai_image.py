from pathlib import Path
import gc
import os

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "input.png"
prompt_path = BASE_DIR / "prompt.txt"
output_path = BASE_DIR / "output.png"

prompt = prompt_path.read_text(encoding="utf-8").strip()

original = Image.open(input_path).convert("RGB")
original_size = original.size

print("Loading old Stable Diffusion...")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    dtype=torch.float16
).to("cuda")

print("Running old AI...")

w, h = original.size

# Stable Diffusion works with dimensions divisible by 8.
w = max(64, (w // 8) * 8)
h = max(64, (h // 8) * 8)

working = original.resize((w, h))

result = pipe(
    prompt=prompt,
    image=working,
    strength=1,
    guidance_scale=7.5,
    num_inference_steps=10
).images[0]

# Restore the exact original dimensions.
result = result.resize(original_size)

# Close the input image before saving.
original.close()

# Delete an existing output file if possible.
if output_path.exists():
    try:
        output_path.unlink()
    except PermissionError:
        print("WARNING: output.png is currently being used by another program.")
        print("Close anything viewing output.png and try again.")
        raise

# Save explicitly.
print(f"Saving to: {output_path}")

with open(output_path, "wb") as f:
    result.save(f, format="PNG")

print("Successfully saved output.png!")

del result
del pipe
gc.collect()
torch.cuda.empty_cache()