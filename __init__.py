import contextlib
import random
from collections import Counter

import numpy as np
import torch
from compel import Compel, DiffusersTextualInversionManager
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

import comfy.model_management
import folder_paths
from comfy.utils import ProgressBar

from .pipelines import PipelineWrapper, schedulers


def get_prompt_embeds(pipe, prompt, negative_prompt):
    textual_inversion_manager = DiffusersTextualInversionManager(pipe)
    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        textual_inversion_manager=textual_inversion_manager,
        truncate_long_prompts=False,
    )

    prompt_embeds = compel.build_conditioning_tensor(prompt)
    negative_prompt_embeds = compel.build_conditioning_tensor(negative_prompt)
    [
        prompt_embeds,
        negative_prompt_embeds,
    ] = compel.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, negative_prompt_embeds]
    )
    return prompt_embeds, negative_prompt_embeds


def latents_to_tensor(pipeline, latents):
    image_numpy = pipeline.decode_latents(latents)  # numpy
    return torch.tensor(image_numpy)


def prepare_latents(
    pipe: StableDiffusionPipeline,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    latents=None,
):
    shape = (
        batch_size,
        num_channels_latents,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * pipe.scheduler.init_noise_sigma
    return latents


class GetFilledColorImage:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Jannchie"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "red": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "green": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "blue": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
            },
        }

    def run(self, width, height, red, green, blue):
        image = torch.tensor(np.full((height, width, 3), (red, green, blue)))
        # 再转换成 0 - 1 之间的浮点数
        image = image
        image = image.unsqueeze(0)
        return (image,)


class DiffusersCompelPromptEmbedding:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("DIFFUSERS_PROMPT_EMBEDDING", "DIFFUSERS_PROMPT_EMBEDDING")
    RETURN_NAMES = ("positive prompt embedding", "negative prompt embedding")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFUSERS_PIPELINE",),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "(masterpiece)1.2, (best quality)1.4",
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def run(
        self,
        pipeline: StableDiffusionPipeline,
        positive_prompt: str,
        negative_prompt: str,
    ):
        return get_prompt_embeds(pipeline, positive_prompt, negative_prompt)


class DiffusersTextureInversionLoader:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFUSERS_PIPELINE",),
                "texture_inversion": (folder_paths.get_filename_list("embeddings"),),
            },
        }

    def run(self, pipeline: StableDiffusionPipeline, texture_inversion: str):
        with contextlib.suppress(Exception):
            path = folder_paths.get_full_path("embeddings", texture_inversion)
            token = texture_inversion.split(".")[0]
            pipeline.load_textual_inversion(path, token=token)
        return (pipeline,)


class GetAverageColorFromImage:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("red", "green", "blue")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "average": ("STRING", {"default": "mean", "options": ["mean", "mode"]}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def run(self, image: torch.Tensor, average: str, mask: torch.Tensor = None):
        if average == "mean":
            return self.run_avg(image, mask)
        elif average == "mode":
            return self.run_mode(image, mask)

    def run_avg(self, image: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        masked_image = image * mask if mask is not None else image
        pixel_sum = torch.sum(masked_image, dim=(2, 3))
        pixel_count = (
            torch.sum(mask, dim=(2, 3))
            if mask is not None
            else torch.prod(torch.tensor(image.shape[2:]))
        )
        average_rgb = pixel_sum / pixel_count.unsqueeze(1)

        average_rgb = torch.round(average_rgb)

        return tuple(average_rgb.squeeze().tolist())

    def run_mode(self, image: torch.Tensor, mask: torch.Tensor = None):
        image = image.permute(0, 3, 1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        masked_image = image * mask if mask is not None else image
        pixel_values = masked_image.view(
            masked_image.shape[0], masked_image.shape[1], -1
        )
        pixel_values = pixel_values.permute(0, 2, 1)
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[2])
        pixel_values = [
            tuple(color.tolist()) for color in pixel_values.numpy() if color.max() > 0
        ]

        if not pixel_values:
            return (0, 0, 0)

        color_counts = Counter(pixel_values)

        return max(color_counts, key=color_counts.get)


class DiffusersPipeline:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "vae_name": (
                    folder_paths.get_filename_list("vae") + ["-"],
                    {"default": "-"},
                ),
                "scheduler_name": (
                    list(schedulers.keys()) + ["-"],
                    {
                        "default": "-",
                    },
                ),
            },
        }

    def run(self, ckpt_name: str, vae_name: str = None, scheduler_name: str = None):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if vae_name == "-":
            vae_path = None
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
        if scheduler_name == "-":
            scheduler_name = None

        self.pipeline_wrapper = PipelineWrapper(ckpt_path, vae_path, scheduler_name)
        return (self.pipeline_wrapper.pipeline,)


class DiffusersPrepareLatents:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFUSERS_PIPELINE",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 64}),
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 64}),
            },
            "optional": {
                "latents": ("LATENT", {"default": None}),
                "seed": (
                    "INT",
                    {"default": None, "min": 0, "step": 1, "max": 999999999},
                ),
            },
        }

    def run(
        self,
        pipeline: StableDiffusionPipeline,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        latents: torch.Tensor | None = None,
        seed: int | None = None,
    ):
        if seed is None:
            seed = random.randint(0, 999999999)
        device = comfy.model_management.get_torch_device()
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        latents = prepare_latents(
            pipe=pipeline,
            batch_size=batch_size,
            num_channels_latents=4,
            height=height,
            width=width,
            dtype=comfy.model_management.VAE_DTYPE,
            device=device,
            generator=generator,
            latents=latents,
        )
        return (latents,)


class DiffusersDecoder:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFUSERS_PIPELINE",),
                "latents": ("LATENT",),
            },
        }

    def run(self, pipeline: StableDiffusionPipeline, latents: torch.Tensor):
        return (latents_to_tensor(pipeline, latents),)


class DiffusersGenerate:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFUSERS_PIPELINE",),
                "positive_prompt_embedding": ("DIFFUSERS_PROMPT_EMBEDDING",),
                "negative_prompt_embedding": ("DIFFUSERS_PROMPT_EMBEDDING",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.02},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 30, "min": 1, "max": 100, "step": 1},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1, "max": 999999999999},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                    },
                ),
            },
        }

    def run(
        self,
        pipeline: StableDiffusionPipeline,
        positive_prompt_embedding: torch.Tensor,
        negative_prompt_embedding: torch.Tensor,
        width: int,
        height: int,
        batch_size: int,
        images: torch.Tensor | None = None,
        num_inference_steps: int = 30,
        strength: float = 1.0,
        seed=None,
    ):
        pbar = ProgressBar(int(num_inference_steps * strength))
        device = comfy.model_management.get_torch_device()
        if not seed:
            seed = random.randint(0, 999999999999)
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        # (B, H, W, C) to (B, C, H, W)
        if images is None:
            latents = prepare_latents(
                pipe=pipeline,
                batch_size=batch_size,
                num_channels_latents=4,
                height=height,
                width=width,
                generator=generator,
                dtype=comfy.model_management.VAE_DTYPE,
                device=device,
            )
            images = latents_to_tensor(pipeline, latents).permute(0, 3, 1, 2)
        else:
            images = images.permute(0, 3, 1, 2)
        # positive_prompt_embedding 和 negative_prompt_embedding 需要匹配 batch_size
        positive_prompt_embedding = positive_prompt_embedding.repeat(batch_size, 1, 1)
        negative_prompt_embedding = negative_prompt_embedding.repeat(batch_size, 1, 1)

        def callback(*_):
            pbar.update(1)

        result = pipeline(
            image=images,
            generator=generator,
            prompt_embeds=positive_prompt_embedding,
            negative_prompt_embeds=negative_prompt_embedding,
            num_inference_steps=num_inference_steps,
            callback_steps=1,
            strength=strength,
            callback=callback,
            return_dict=True,
        )
        # image = result["images"][0]
        # images to torch.Tensor
        imgs = [np.array(img) for img in result["images"]]
        imgs = torch.tensor(imgs, dtype=images.dtype)
        result["images"][0].save("1.png")
        # 0 ~ 255 to 0 ~ 1
        imgs = imgs / 255
        # (B, C, H, W) to (B, H, W, C)
        return (imgs,)


NODE_CLASS_MAPPINGS = {
    "GetFilledColorImage": GetFilledColorImage,
    "GetAverageColorFromImage": GetAverageColorFromImage,
    "DiffusersPipeline": DiffusersPipeline,
    "DiffusersGenerate": DiffusersGenerate,
    "DiffusersPrepareLatents": DiffusersPrepareLatents,
    "DiffusersDecoder": DiffusersDecoder,
    "DiffusersCompelPromptEmbedding": DiffusersCompelPromptEmbedding,
    "DiffusersTextureInversionLoader": DiffusersTextureInversionLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetFilledColorImage": "Get Filled Color Image Jannchie",
    "GetAverageColorFromImage": "Get Average Color From Image Jannchie",
    "DiffusersPipeline": "Diffusers Pipeline",
    "DiffusersGenerate": "Diffusers Generate",
    "DiffusersPrepareLatents": "Diffusers Prepare Latents",
    "DiffusersDecoder": "Diffusers Decoder",
    "DiffusersCompelPromptEmbedding": "Diffusers Compel Prompt Embedding",
    "DiffusersTextureInversionLoader": "Diffusers Texture Inversion Embedding Loader",
}
