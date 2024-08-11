import io
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI, Response, HTTPException

import models

model_functions = {
    "stable_diffusion_2_1": models.stable_diffusion_2_1.generate_image,
    "flux_1": models.flux_1.generate_image
}

app = FastAPI()

@app.get("/generate")
def generate(model: str, prompt: str):
    
    if model not in model_functions:
        available_models = list(model_functions.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model not found. Available models are: {', '.join(available_models)}",
        )

    image_content = model_functions[model](prompt)

    return Response(content=image_content, media_type="image/png")