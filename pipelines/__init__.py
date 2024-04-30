import contextlib

from diffusers import AutoencoderKL, AutoencoderTiny, DPMSolverMultistepScheduler
from diffusers.schedulers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

import comfy.model_management
import folder_paths

from .jannchie import *

schedulers = {
    "DPM++ 2M": DPMSolverMultistepScheduler(),
    "DPM++ 2M Karras": DPMSolverMultistepScheduler(use_karras_sigmas=True),
    "DPM++ 2M SDE": DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++"),
    "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler(
        use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
    ),
    "DPM++ SDE": DPMSolverSinglestepScheduler(),
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler(use_karras_sigmas=True),
    "DPM2": KDPM2DiscreteScheduler(),
    "DPM2 Karras": KDPM2DiscreteScheduler(use_karras_sigmas=True),
    "DPM2 a": KDPM2AncestralDiscreteScheduler(),
    "DPM2 a Karras": KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True),
    "Euler": EulerDiscreteScheduler(),
    "Euler a": EulerAncestralDiscreteScheduler(),
    "Heun": HeunDiscreteScheduler(),
    "LMS": LMSDiscreteScheduler(),
    "LMS Karras": LMSDiscreteScheduler(use_karras_sigmas=True),
    "DEIS": DEISMultistepScheduler(),
    "UniPC": UniPCMultistepScheduler(),
}


class PipelineWrapper:

    def __init__(
        self,
        ckpt_path: str,
        vae_path: str = None,
        scheduler_name: str = None,
        use_tiny_vae: bool = False,
    ):
        scheduler = schedulers.get(scheduler_name)
        device = comfy.model_management.get_torch_device()
        vae_dtype = comfy.model_management.vae_dtype()
        unet_dtype = comfy.model_management.unet_dtype()
        if ckpt_path.endswith(".safetensors"):
            self.pipeline = JannchiePipeline.from_single_file(
                ckpt_path,
                torch_dtype=unet_dtype,
                cache_dir=folder_paths.get_folder_paths("diffusers"),
                use_safetensors=True,
            )
        else:
            self.pipeline = JannchiePipeline.from_pretrained(
                ckpt_path,
                torch_dtype=unet_dtype,
                cache_dir=folder_paths.get_folder_paths("diffusers"),
                use_safetensors=ckpt_path.endswith(".safetensors"),
            )

        if use_tiny_vae:
            self.pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                device=self.pipeline.device, dtype=vae_dtype
            )

        elif vae_path:
            if vae_path.endswith(".safetensors"):
                self.pipeline.vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=vae_dtype,
                    cache_dir=folder_paths.get_folder_paths("diffusers"),
                    use_safetensors=True,
                )
            else:
                self.pipeline.vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    torch_dtype=vae_dtype,
                    cache_dir=folder_paths.get_folder_paths("diffusers"),
                    use_safetensors=vae_path.endswith(".safetensors"),
                )

        if scheduler:
            self.pipeline.scheduler = scheduler
        self.pipeline.to(device)
        self.pipeline.vae.to(vae_dtype)
        self.pipeline.safety_checker = None
        with contextlib.suppress(Exception):
            self.pipeline.enable_xformers_memory_efficient_attention()
