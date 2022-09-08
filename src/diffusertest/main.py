# make sure you're logged in with `huggingface-cli login`
import os

from diffusers import StableDiffusionPipeline

token = "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"
token = os.environ.get("HUGGING_FACE_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=token)
pipe = pipe.to("mps")

prompt = "a photo of an astronaut riding a horse on mars"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]
