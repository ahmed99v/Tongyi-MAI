import argparse
import sys
from pathlib import Path

import torch
from diffusers import ZImagePipeline

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_OUTPUT = "example.png"

# --------------------------------------------------
# Device helpers
# --------------------------------------------------
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. Install CUDA-enabled PyTorch and ensure an NVIDIA GPU is available."
        )
    return torch.device("cuda")


def get_dtype(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# --------------------------------------------------
# Pipeline
# --------------------------------------------------
def load_pipeline(device):
    dtype = get_dtype(device)
    pipe = ZImagePipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return pipe.to(device)


# --------------------------------------------------
# Generator
# --------------------------------------------------
def create_generator(device, seed=42):
    generator = torch.Generator(device=device)
    return generator.manual_seed(seed)

# --------------------------------------------------
# Image generation
# --------------------------------------------------
def run_generation(pipe, prompt, generator):
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    )
    return result.images[0]

def save_image(image, path):
    path = Path(path)
    image.save(path)
    print(f"Image saved to {path.resolve()}")

# ---------------CLI------------------# 
def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Generate an image with Tongyi-MAI Z-Image-Turbo"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output image file",
    )
    return parser.parse_args(argv)


# Main workflow #
def main(argv=None):
    args = parse_args(argv)
    device = get_device()
    pipe = load_pipeline(device)
    generator = create_generator(device)
    image = run_generation(
        pipe=pipe,
        prompt=args.prompt,
        generator=generator,
    )
    save_image(image, args.output)

if __name__ == "__main__":
    main(sys.argv[1:])