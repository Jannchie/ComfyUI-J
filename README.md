# ComfyUI-J

Jannchie's ComfyUI custom nodes.

This is a completely different set of nodes than Comfy's own KSampler series.
This set of nodes is based on Diffusers, which makes it easier to import models, apply prompts with weights, inpaint, reference only, controlnet, etc.

## Installation

In the `custom_nodes` directory, run

```bash
git clone github.com/Jannchie/ComfyUI-J
cd ComfyUI-J
pip install -r requirements.txt
```

## Examples

### Base Usage of Jannchie's Diffusers Pipeline

![Base Usage](./examples/base.png)

### Reference Only with Jannchie's Diffusers Pipeline

![Reference only](./examples/reference_only.png)

### ControlNet with Jannchie's Diffusers Pipeline

![ControlNet](./examples/controlnet.png)

## Inpainting with Jannchie's Diffusers Pipeline

![Inpainting](./examples/inpainting.png)

## Remove something with Jannchie's Diffusers Pipeline

![Remove something](./examples/remove_something.png)

## Change Clothes with Jannchie's Diffusers Pipeline

This is a composite application of diffusers pipeline custom node. Includes:

- Reference only
- ControlNet
- Inpainting
- Texture Inversion

This is a demonstration of a simple workflow for properly dressing a character.

A checkpoint for stablediffusion 1.5 is all your need. But for full automation, I use the Comfyui_segformer_b2_clothes custom node for generating masks. you can draw your own masks without it.

![Change Clothes](./examples/change_clothes.png)

## TODO

- [ ] Add LoRA support
