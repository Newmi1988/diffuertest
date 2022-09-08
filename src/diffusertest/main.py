# make sure you're logged in with `huggingface-cli login`
import os
from pathlib import Path

from diffusers import StableDiffusionPipeline

token = os.environ.get("HUGGING_FACE_TOKEN")
output_dir = Path("./artefacts")
output_dir.mkdir(parents=True,exist_ok=True)
output_dir = output_dir.resolve().absolute()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=token)
pipe = pipe.to("mps")

prompt = "a painting of a cat in an uniform"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
images = pipe(prompt).images
for image in images:
    prompt_combined = prompt.replace(" ","_")
    output_dir_string = str(output_dir)
    image.save(f"{output_dir_string}/{prompt_combined}.png")
