import contextlib

from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
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
        self, ckpt_path: str, vae_path: str = None, scheduler_name: str = None
    ):
        scheduler = schedulers.get(scheduler_name)
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.VAE_DTYPE
        if ckpt_path.endswith(".safetensors"):
            self.pipeline = JannchiePipeline.from_single_file(
                ckpt_path,
                torch_dtype=dtype,
            )
        else:
            self.pipeline = JannchiePipeline.from_pretrained(
                ckpt_path,
                torch_dtype=dtype,
            )
        if vae_path:
            if vae_path.endswith(".safetensors"):
                self.pipeline.vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=dtype,
                )
            else:
                self.pipeline.vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    torch_dtype=dtype,
                )
        if scheduler:
            self.pipeline.scheduler = scheduler
        self.pipeline.to(device)
        self.pipeline.safety_checker = None
        with contextlib.suppress(Exception):
            self.pipeline.enable_xformers_memory_efficient_attention()
