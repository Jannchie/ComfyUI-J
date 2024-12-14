import contextlib
import gc
import random
from collections import Counter

import numpy as np
import torch
from compel import Compel, DiffusersTextualInversionManager
from diffusers import StableDiffusionPipeline
from diffusers.models import ControlNetModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

import comfy.model_management
import folder_paths
from comfy.utils import ProgressBar

from .pipelines import ControlNetUnit, ControlNetUnits, PipelineWrapper, schedulers

from configs.node_fields import get_field_pre_values



def resize_with_padding(image: Image.Image, target_size: tuple[int, int]):
    # 打开图像

    # 计算缩放比例
    width_ratio = target_size[0] / image.width
    height_ratio = target_size[1] / image.height
    ratio = min(width_ratio, height_ratio)

    # 计算调整后的尺寸
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    # 缩放图像
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建黑色背景图像
    background = Image.new("RGBA", target_size, (0, 0, 0, 0))

    # 计算粘贴位置
    position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)

    # 粘贴调整后的图像到黑色背景上
    background.paste(image, position)
    return background


def comfy_image_to_pil(image: torch.Tensor):
    image = image.squeeze(0)  # (1, H, W, C) => (H, W, C)
    image = image * 255  # 0 ~ 1 => 0 ~ 255
    image = image.to(dtype=torch.uint8)  # float32 => uint8
    return Image.fromarray(image.numpy())  # tensor => PIL.Image.Image


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


def latents_to_img_tensor(pipeline, latents):
    # 1. 输入的 latents 是一个 -1 ~ 1 之间的 tensor
    # 2. 先进行缩放
    scaled_latents = latents / pipeline.vae.config.scaling_factor
    # 转成 vae 类型
    scaled_latents = scaled_latents.to(dtype=comfy.model_management.vae_dtype())
    print(scaled_latents.dtype, pipeline.vae.dtype)
    # 3. 解码，返回的是 -1 ~ 1 之间的 tensor
    dec_tensor = pipeline.vae.decode(scaled_latents, return_dict=False)[0]
    # 4. 缩放到 0 ~ 1 之间
    dec_images = pipeline.image_processor.postprocess(
        dec_tensor,
        output_type="pt",
        do_denormalize=[True for _ in range(scaled_latents.shape[0])],
    )
    # 5. 转换成 tensor,
    res = torch.nan_to_num(dec_images).to(dtype=torch.float32)
    # 6. 将 channel 放到最后
    # res shape torch.Size([1, 3, 512, 512]) => torch.Size([1, 512, 512, 3])
    res = res.permute(0, 2, 3, 1)
    return res


def latents_to_mask_tensor(pipeline, latents):
    # 1. 输入的 latents 是一个 -1 ~ 1 之间的 tensor
    # 2. 先进行缩放
    scaled_latents = latents / pipeline.vae.config.scaling_factor
    # 3. 解码，返回的是 -1 ~ 1 之间的 tensor
    dec_tensor = pipeline.vae.decode(scaled_latents, return_dict=False)[0]
    # 4. 缩放到 0 ~ 1 之间
    dec_images = pipeline.mask_processor.postprocess(
        dec_tensor,
        output_type="pt",
    )
    # 5. 转换成 tensor,
    res = torch.nan_to_num(dec_images).to(dtype=torch.float32)
    # 6. 将 channel 放到最后
    # res shape torch.Size([1, 3, 512, 512]) => torch.Size([1, 512, 512, 3])
    res = res.permute(0, 2, 3, 1)
    return res


def prepare_latents(
    pipe: StableDiffusionPipeline,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    latents=None,
):
    shape = (
        batch_size,
        pipe.unet.config.in_channels,
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


def prepare_image(
    pipeline: StableDiffusionPipeline,
    seed=47,
    batch_size=1,
    height=512,
    width=512,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    latents = prepare_latents(
        pipe=pipeline,
        batch_size=batch_size,
        height=height,
        width=width,
        generator=generator,
        device=comfy.model_management.get_torch_device(),
        dtype=comfy.model_management.VAE_DTYPE,
    )
    return latents_to_img_tensor(pipeline, latents)


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
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "green": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "blue": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
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
                #liblib adapter 为了防止节点注释掉导致无法找到，这里将返回空的
                # "texture_inversion": (folder_paths.get_filename_list("embeddings"),),
                "texture_inversion": ([],)
            },
        }

    def run(self, pipeline: StableDiffusionPipeline, texture_inversion: str):
        with contextlib.suppress(Exception):
            path = folder_paths.get_full_path("embeddings", texture_inversion)
            token = texture_inversion.split(".")[0]
            pipeline.load_textual_inversion(path, token=token)
            print(f"Loaded {texture_inversion}")
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
                "average": (("mean", "mode"),),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def run(self, image: torch.Tensor, average: str, mask: torch.Tensor = None):
        if mask is not None:
            assert (
                mask.ndim == image.ndim - 1
            ), "Mask dimensions must be one less than image dimensions."
            mask = mask.unsqueeze(3)  # Unsqueeze to match (B, 1, H, W)
        if mask is not None and torch.sum(mask) == 0:
            mask = None
        if average == "mean":
            return self.run_avg(image, mask)
        elif average == "mode":
            return self.run_mode(image, mask)
        else:
            raise ValueError("average must be either 'mean' or 'mode'")

    def run_avg(self, image: torch.Tensor, mask: torch.Tensor = None):
        masked_image = image * mask if mask is not None else image

        pixel_sum = torch.sum(masked_image, dim=(1, 2))
        if mask is not None:
            pixel_count = torch.sum(mask, dim=(1, 2)).unsqueeze(1)
        else:
            pixel_count = torch.tensor(image.shape[1] * image.shape[2]).unsqueeze(0)
        average_rgb = pixel_sum / pixel_count
        average_rgb = torch.round(average_rgb * 255)
        return tuple(average_rgb.squeeze().int().tolist())

    def run_mode(self, image: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            image = image * mask

        # Flatten the image to a 2D matrix where each row is a color
        flattened_image = image.view(-1, image.shape[-1])

        # If mask is provided, remove rows where mask is zero
        if mask is not None:
            flattened_mask = mask.view(-1, 1)
            flattened_image = flattened_image[flattened_mask.squeeze() > 0]

        # Convert the pixel values to a format that can be efficiently counted
        unique_colors, counts = torch.unique(flattened_image, return_counts=True, dim=0)

        # Find the most frequent color
        max_idx = torch.argmax(counts)
        mode_rgb = unique_colors[max_idx]

        mode_rgb = torch.round(mode_rgb * 255)
        return tuple(mode_rgb.int().tolist())


class DiffusersXLPipeline:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ([],),
            },
            "optional": {
                "vae_name": (
                    get_field_pre_values("DiffusersXLPipeline", "vae_name") + ["-"],
                    # folder_paths.get_filename_list("vae") + ["-"],
                    {"default": "-"},
                ),
                "scheduler_name": (
                    list(schedulers.keys()) + ["-"],
                    {
                        "default": "-",
                    },
                ),
                "use_tiny_vae": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                    },
                ),
            },
        }

    def run(
        self,
        ckpt_name: str,
        vae_name: str = None,
        scheduler_name: str = None,
        use_tiny_vae: str = "disable",
    ):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path is None:
            ckpt_path = ckpt_name
        if vae_name == "-":
            vae_path = None
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
        if scheduler_name == "-":
            scheduler_name = None

        self.pipeline_wrapper = PipelineWrapper(
            ckpt_path,
            vae_path,
            scheduler_name,
            pipeline=StableDiffusionPipeline,
            use_tiny_vae=use_tiny_vae == "enable",
        )
        return (self.pipeline_wrapper.pipeline,)


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
                    get_field_pre_values("DiffusersPipeline", "vae_name") + ["-"],
                    # folder_paths.get_filename_list("vae") + ["-"],
                    {"default": "-"},
                ),
                "scheduler_name": (
                    list(schedulers.keys()) + ["-"],
                    {
                        "default": "-",
                    },
                ),
                "use_tiny_vae": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                    },
                ),
            },
        }

    def run(
        self,
        ckpt_name: str,
        vae_name: str = None,
        scheduler_name: str = None,
        use_tiny_vae: str = "disable",
    ):
        torch.cuda.empty_cache()
        gc.collect()
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path is None:
            ckpt_path = ckpt_name
        if vae_name == "-":
            vae_path = None
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
        if scheduler_name == "-":
            scheduler_name = None

        self.pipeline_wrapper = PipelineWrapper(
            ckpt_path, vae_path, scheduler_name, use_tiny_vae=use_tiny_vae == "enable"
        )
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
            height=height,
            width=width,
            dtype=comfy.model_management.vae_dtype(),
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
        res = latents_to_img_tensor(pipeline, latents)
        return (res,)


# 'https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth'

controlnet_list = [
    "canny",
    "openpose",
    "depth",
    "tile",
    "ip2p",
    "shuffle",
    "inpaint",
    "lineart",
    "mlsd",
    "normalbae",
    "scribble",
    "seg",
    "softedge",
    "lineart_anime",
    "other",
]


class DiffusersControlNetLoader:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("DIFFUSERS_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #liblib adapter 
                # "controlnet_model_name": (controlnet_list,),
                "controlnet_model_name": (get_field_pre_values("DiffusersControlnetLoader", "controlnet_model_name"),)
            },
            "optional": {
                #liblib adapter 不支持其他的，因为没有哦可见列表
                # "controlnet_model_file": (folder_paths.get_filename_list("controlnet"),)
                "controlnet_model_file": (["None"],)
            },
        }

    def run(self, controlnet_model_name: str, controlnet_model_file: str = ""):
        #liblib adapter 
        # file_list = folder_paths.get_filename_list("controlnet")
        # if controlnet_model_name == "other":
        #     controlnet_model_path = folder_paths.get_full_path(
        #         "controlnet", controlnet_model_file
        #     )
        # else:
        #     if controlnet_model_name == "depth":
        #         file_name = f"control_v11f1p_sd15_{controlnet_model_name}.pth"
        #     elif controlnet_model_name == "tile":
        #         file_name = f"control_v11f1e_sd15_{controlnet_model_name}.pth"
        #     else:
        #         file_name = f"control_v11p_sd15_{controlnet_model_name}.pth"
        #     controlnet_model_path = next(
        #         (
        #             folder_paths.get_full_path("controlnet", file)
        #             for file in file_list
        #             if file_name in file
        #         ),
        #         None,
        #     )
        # if controlnet_model_path is None:
        #     controlnet_model_path = f"https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/{file_name}"
        print("controlnet_model_name:", controlnet_model_name)
        controlnet_model_path = controlnet_model_name
        controlnet = ControlNetModel.from_single_file(
            controlnet_model_path,
            cache_dir=folder_paths.get_folder_paths("controlnet")[0],
        ).to(
            device=comfy.model_management.get_torch_device(),
            dtype=comfy.model_management.unet_dtype(),
        )
        return (controlnet,)


class DiffusersControlNetUnit:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("CONTROLNET_UNIT",)
    RETURN_NAMES = ("controlnet unit",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("DIFFUSERS_CONTROLNET",),
                "image": ("IMAGE",),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
            },
        }

    def run(
        self,
        controlnet: ControlNetModel,
        image: torch.Tensor,
        scale: float,
        start: float,
        end: float,
    ):
        unit = ControlNetUnit(
            controlnet=controlnet,
            image=comfy_image_to_pil(image),
            scale=scale,
            start=start,
            end=end,
        )
        return ((unit,),)


class DiffusersControlNetUnitStack:
    CATEGORY = "Jannchie"
    FUNCTION = "run"
    RETURN_TYPES = ("CONTROLNET_UNIT",)
    RETURN_NAMES = ("controlnet unit",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_unit_1": ("CONTROLNET_UNIT",),
            },
            "optional": {
                "controlnet_unit_2": (
                    "CONTROLNET_UNIT",
                    {
                        "default": None,
                    },
                ),
                "controlnet_unit_3": (
                    "CONTROLNET_UNIT",
                    {
                        "default": None,
                    },
                ),
            },
        }

    def run(
        self,
        controlnet_unit_1: tuple[ControlNetModel],
        controlnet_unit_2: tuple[ControlNetModel] | None = None,
        controlnet_unit_3: tuple[ControlNetModel] | None = None,
    ):
        stack = []
        if controlnet_unit_1:
            stack += controlnet_unit_1
        if controlnet_unit_2:
            stack += controlnet_unit_2
        if controlnet_unit_3:
            stack += controlnet_unit_3
        return (stack,)


class DiffusersGenerator:
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
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.02},
                ),
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
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "reference_style_fidelity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "controlnet_units": ("CONTROLNET_UNIT",),
                "reference_image": (
                    "IMAGE",
                    {"default": None},
                ),
                "reference_only": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                    },
                ),
                "reference_only_adain": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
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
        guidance_scale: float = 7.0,
        controlnet_units: tuple[ControlNetUnit] = None,
        seed=None,
        mask: torch.Tensor | None = None,
        reference_only: str = "disable",
        reference_only_adain: str = "disable",
        reference_image: torch.Tensor | None = None,
        reference_style_fidelity: float = 0.5,
        reference_strength: float = 1.0,
    ):
        reference_only = reference_only == "enable"
        reference_only_adain = reference_only_adain == "enable"
        latents = None
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
                height=height,
                width=width,
                generator=generator,
                device=device,
                dtype=comfy.model_management.vae_dtype(),
            )
            images = latents_to_img_tensor(pipeline, latents)
        else:
            images = images

        # positive_prompt_embedding 和 negative_prompt_embedding 需要匹配 batch_size
        positive_prompt_embedding = positive_prompt_embedding.repeat(batch_size, 1, 1)
        negative_prompt_embedding = negative_prompt_embedding.repeat(batch_size, 1, 1)
        width = images.shape[2]
        height = images.shape[1]

        def callback(*_):
            pbar.update(1)

        if controlnet_units is not None:
            for unit in controlnet_units:
                target_image_shape = (width, height)
                unit_img = resize_with_padding(unit.image, target_image_shape)
                unit.image = unit_img
            controlnet_units = ControlNetUnits(controlnet_units)
        result = pipeline(
            image=images,
            mask_image=mask,
            ref_image=reference_image if reference_image is not None else images,
            generator=generator,
            width=width,
            height=height,
            prompt_embeds=positive_prompt_embedding,
            negative_prompt_embeds=negative_prompt_embedding,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback_steps=1,
            strength=strength,
            controlnet_units=controlnet_units,
            callback=callback,
            reference_strength=reference_strength,
            reference_attn=reference_only,
            reference_adain=reference_only_adain,
            style_fidelity=reference_style_fidelity,
            return_dict=True,
        )
        # image = result["images"][0]
        # images to torch.Tensor
        imgs = [np.array(img) for img in result["images"]]
        imgs = torch.tensor(imgs)
        result["images"][0].save("1.png")
        # 0 ~ 255 to 0 ~ 1
        imgs = imgs / 255
        # (B, C, H, W) to (B, H, W, C)
        torch.cuda.empty_cache()
        gc.collect()
        return (imgs,)


NODE_CLASS_MAPPINGS = {
    "GetFilledColorImage": GetFilledColorImage,
    "GetAverageColorFromImage": GetAverageColorFromImage,
    "DiffusersPipeline": DiffusersPipeline,
    "DiffusersXLPipeline": DiffusersXLPipeline,
    "DiffusersGenerator": DiffusersGenerator,
    "DiffusersPrepareLatents": DiffusersPrepareLatents,
    "DiffusersDecoder": DiffusersDecoder,
    "DiffusersCompelPromptEmbedding": DiffusersCompelPromptEmbedding,
    "DiffusersTextureInversionLoader": DiffusersTextureInversionLoader,
    "DiffusersControlnetLoader": DiffusersControlNetLoader,
    "DiffusersControlnetUnit": DiffusersControlNetUnit,
    "DiffusersControlnetUnitStack": DiffusersControlNetUnitStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetFilledColorImage": "Get Filled Color Image Jannchie",
    "GetAverageColorFromImage": "Get Average Color From Image Jannchie",
    "DiffusersPipeline": "🤗 Diffusers Pipeline",
    "DiffusersXLPipeline": "🤗 Diffusers XL Pipeline",
    "DiffusersGenerator": "🤗 Diffusers Generator",
    "DiffusersPrepareLatents": "🤗 Diffusers Prepare Latents",
    "DiffusersDecoder": "🤗 Diffusers Decoder",
    "DiffusersCompelPromptEmbedding": "🤗 Diffusers Compel Prompt Embedding",
    "DiffusersTextureInversionLoader": "🤗 Diffusers Texture Inversion Embedding Loader",
    "DiffusersControlnetLoader": "🤗 Diffusers Controlnet Loader",
    "DiffusersControlnetUnit": "🤗 Diffusers Controlnet Unit",
    "DiffusersControlnetUnitStack": "🤗 Diffusers Controlnet Unit Stack",
}
