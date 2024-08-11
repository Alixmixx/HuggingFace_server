import torch
import io
from diffusers import FluxPipeline

def generate_image(prompt: str):
    
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    image_store = io.BytesIO()
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save(image_store, "PNG")
    
    return image_store.getvalue()