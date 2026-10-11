from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

prompt = input("Enter a prompt: ")

# Save the prompt for old_ai_image.py
(BASE_DIR / "prompt.txt").write_text(
    prompt,
    encoding="utf-8"
)

# Run the text-to-image stage in a completely separate process.
subprocess.run(
    [
        sys.executable,
        "-c",
        r'''
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

base = Path.cwd()

prompt = ''' + repr(prompt) + r'''

print("Loading Stable Diffusion...")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    dtype=torch.float16
).to("cuda")

print("Generating input.png...")

image = pipe(
    prompt,
    num_inference_steps=5,
    guidance_scale=4.5
).images[0]

image.save(base / "input.png")

print("Saved input.png")
'''
    ],
    cwd=BASE_DIR,
    check=True
)

print("\nFirst AI process has completely exited.")
print("Starting old AI...\n")

# Now start old_ai_image.py AFTER the first process is dead.
subprocess.run(
    [sys.executable, str(BASE_DIR / "old_ai_image.py")],
    cwd=BASE_DIR,
    check=True
)

print("\nDone!")