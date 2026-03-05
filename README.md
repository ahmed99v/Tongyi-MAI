# Z-Image Turbo Image Generation Script

This project contains a Python script that generates images from text prompts using the **Tongyi-MAI Z-Image-Turbo** model through the **diffusers** library.
The script loads the model, processes a text prompt, generates an image, and saves the output file locally.

The code is organized in a **functional structure**, where each function performs a single responsibility such as loading the model, generating images, or handling command-line input.

---

# Requirements

## Hardware

* NVIDIA GPU with CUDA support

## Software

* Python 3.9 or newer
* CUDA-enabled PyTorch

## Python Dependencies

Install the required libraries:

```bash
pip install torch diffusers transformers accelerate
```

If CUDA-enabled PyTorch is required, install it from:

https://pytorch.org/get-started/locally/

---

# How to Run

Run the script from the command line and provide a prompt.

Example:

```bash
python generate_image.py --prompt "A futuristic city with neon lights" --output result.png
```

Arguments:

| Argument   | Description                            |
| ---------- | -------------------------------------- |
| `--prompt` | Text prompt used to generate the image |
| `--output` | Output image file path                 |

Example command:

```bash
python generate_image.py \
--prompt "Ancient temple in the mountains at sunrise" \
--output temple.png
```

After execution, the generated image will be saved to the specified path.

---

# Script Workflow

The script executes the following steps:

1. Parse command line arguments
2. Detect the available compute device
3. Load the Z-Image-Turbo pipeline
4. Create a random generator for reproducibility
5. Generate an image using the provided prompt
6. Save the generated image

---

# Function Documentation

Below is an explanation of each function used in the script.

---

## `get_device()`

**Purpose**

Detects and returns the available GPU device.

**Behavior**

* Checks whether CUDA is available.
* Raises an error if CUDA is not available because the model requires GPU acceleration.

**Return**

```
torch.device
```

Example return value:

```
cuda
```

---

## `get_dtype(device)`

**Purpose**

Determines the tensor data type used for model execution.

**Behavior**

* If the device is CUDA, it uses `bfloat16` for better performance and reduced memory usage.
* Otherwise it falls back to `float32`.

**Parameters**

| Parameter | Type         | Description                     |
| --------- | ------------ | ------------------------------- |
| device    | torch.device | Device where the model will run |

**Return**

```
torch.dtype
```

---

## `load_pipeline(device)`

**Purpose**

Loads the Z-Image-Turbo model pipeline.

**Behavior**

1. Determines the correct tensor precision
2. Loads the pretrained model
3. Moves the pipeline to the selected device

**Parameters**

| Parameter | Type         | Description                 |
| --------- | ------------ | --------------------------- |
| device    | torch.device | Device used for computation |

**Return**

```
ZImagePipeline
```

---

## `create_generator(device, seed=42)`

**Purpose**

Creates a deterministic random generator.

**Behavior**

* Initializes a PyTorch random generator
* Sets a fixed seed to allow reproducible results

**Parameters**

| Parameter | Type         | Description                    |
| --------- | ------------ | ------------------------------ |
| device    | torch.device | Device where generation occurs |
| seed      | int          | Random seed value              |

**Return**

```
torch.Generator
```

---

## `run_generation(pipe, prompt, generator)`

**Purpose**

Generates an image using the model.

**Behavior**

The function sends a text prompt to the pipeline and runs the diffusion process.

Key parameters used:

| Parameter           | Value |
| ------------------- | ----- |
| height              | 1024  |
| width               | 1024  |
| num_inference_steps | 9     |
| guidance_scale      | 0.0   |

**Parameters**

| Parameter | Type            | Description      |
| --------- | --------------- | ---------------- |
| pipe      | ZImagePipeline  | Loaded pipeline  |
| prompt    | str             | Text prompt      |
| generator | torch.Generator | Random generator |

**Return**

```
PIL.Image
```

---

## `save_image(image, path)`

**Purpose**

Saves the generated image to disk.

**Behavior**

* Converts the path into a `Path` object
* Saves the image
* Prints the final saved location

**Parameters**

| Parameter | Type      | Description      |
| --------- | --------- | ---------------- |
| image     | PIL.Image | Generated image  |
| path      | str       | Output file path |

---

## `parse_args(argv=None)`

**Purpose**

Handles command-line arguments.

**Behavior**

Creates an argument parser that reads:

* `--prompt`
* `--output`

**Return**

```
argparse.Namespace
```

---

## `main(argv=None)`

**Purpose**

Coordinates the entire program workflow.

**Execution Steps**

1. Parse input arguments
2. Select compute device
3. Load the image generation pipeline
4. Create a random generator
5. Generate an image
6. Save the generated image

---

# Generation Parameters

The script uses the following configuration:

| Parameter      | Value       | Description               |
| -------------- | ----------- | ------------------------- |
| image size     | 1024 × 1024 | Output resolution         |
| steps          | 9           | Diffusion steps           |
| guidance scale | 0.0         | Required for Turbo models |
| seed           | 42          | Reproducibility           |

---

# Output

The generated image will be written to the specified output file path.

Example output:

```
Image saved to /your/project/folder/result.png
```

---

# Notes

* CUDA support is required for running the script.
* The `guidance_scale` parameter must be `0.0` when using Turbo models.
* A fixed random seed is used to allow reproducible image generation.
