# Inspired by: https://github.com/Mikubill/sd-webui-controlnet/discussions/1236 and https://github.com/Mikubill/sd-webui-controlnet/discussions/1280
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import ControlNetModel, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

basic_transformer_idx = 0


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _images_to_tensors(
    imgs: List[PIL.Image.Image],
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    buf = []
    for image_ in imgs:
        assert isinstance(image_, PIL.Image.Image)
        image_ = image_.convert("RGB")
        image_ = image_.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)
        image_ = np.array(image_)
        image_ = image_[None, :]
        buf.append(image_)

    image = np.concatenate(buf, axis=0)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    assert isinstance(image, torch.Tensor)

    image = image.to(device=device, dtype=dtype)

    return image


def mask_images_to_float_tensor(
    imgs: List[PIL.Image.Image],
    resize_wh: Optional[Tuple[int, int]] = None,
    resample: Optional[PIL.Image.Resampling] = None,
) -> torch.Tensor:
    width, height = imgs[0].size
    if resize_wh is not None:
        width, height = resize_wh
        if resample is None:
            resample = PIL.Image.Resampling.LANCZOS
        mask = [i.resize((width, height), resample=resample) for i in imgs]
    else:
        mask = imgs
    mask = np.stack([np.array(m.convert("L")) for m in mask], axis=0)
    assert mask.shape == (len(imgs), height, width)
    mask = mask.astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)

    assert mask.shape[0] == len(imgs)
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("Mask should be in [0, 1] range")
    return mask


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


@dataclass
class ControlNetUnit:
    controlnet: ControlNetModel
    image: PIL.Image.Image
    scale: float
    start: float
    end: float


class ControlNetUnits:
    def __init__(
        self,
        units: tuple[ControlNetUnit],
    ):
        self.controlnets = [unit.controlnet for unit in units]
        self.images = [unit.image for unit in units]
        self.scales = [unit.scale for unit in units]
        self.starts = [unit.start for unit in units]
        self.ends = [unit.end for unit in units]


class JannchiePipeline(StableDiffusionControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        ref_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        ref_image_mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_units: ControlNetUnits = None,
        guess_mode: bool = False,
        reference_attn: bool = False,
        reference_adain: bool = False,
        attention_auto_machine_weight: float = 100.0,
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 0.5,
        write_mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        bool_mask=False,
        desc: Optional[str] = None,
        strength=1.0,
        timesteps: List[int] = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: Optional[torch.FloatTensor] = None,
        *arg,
        **args,
    ):
        device = self._execution_device
        if height == None:
            if isinstance(image, torch.Tensor):
                if image is not None:
                    height = image.shape[-2]
                elif ref_image is not None:
                    height = ref_image.shape[-2]
                else:
                    height = 512
            elif isinstance(image, PIL.Image.Image):
                _, height = image.size
        if width == None:
            if isinstance(image, torch.Tensor):
                if image is not None:
                    width = image.shape[-1]
                elif ref_image is not None:
                    width = ref_image.shape[-1]
                else:
                    width = 512
            elif isinstance(image, PIL.Image.Image):
                width, _ = image.size

        if arg or args:
            logger.warning(f"Unused arguments: {arg}, {args}")
        if desc is None:
            desc = "Jannchie's Pipeline"
        self.set_progress_bar_config(desc=desc)
        controlnet_conditioning_scale = []
        control_guidance_start = []
        control_guidance_end = []
        if controlnet_units:
            self.controlnet = MultiControlNetModel(
                controlnets=controlnet_units.controlnets
            )
            controlnet_images = controlnet_units.images
            control_guidance_start = controlnet_units.starts
            control_guidance_end = controlnet_units.ends
            controlnet_conditioning_scale = controlnet_units.scales
        else:
            controlnet_images = []
            self.controlnet = MultiControlNetModel(controlnets=[])
        if not reference_attn and not reference_adain:
            ref_image = None
        if self.controlnet:
            controlnet = (
                self.controlnet._orig_mod
                if is_compiled_module(self.controlnet)
                else self.controlnet
            )
            controlnet.to(device)
            n_controlnet_unit = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 0
            )
        else:
            n_controlnet_unit = 0
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            controlnet_images,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # 输入的是 prompt_embeds
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if self.controlnet:
            if len(self.controlnet.nets) > 1:
                assert isinstance(controlnet_images, list)
            if n_controlnet_unit != 0:
                global_pool_conditions = (
                    controlnet.config.global_pool_conditions
                    if isinstance(controlnet, ControlNetModel)
                    else controlnet.nets[0].config.global_pool_conditions
                )
                guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        logger.debug("Encoding prompt")
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        prompt_embeds = torch.cat(prompt_embeds[::-1], dim=0)

        # 4. Prepare image
        logger.debug("Preparing image")
        if n_controlnet_unit != 0:
            if isinstance(controlnet, ControlNetModel):
                controlnet_images = self.prepare_image(
                    image=controlnet_images,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                ).to(device=device)
                height, width = controlnet_images.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in controlnet_images:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    ).to(device=device)

                    images.append(image_)

                controlnet_images = images
                height, width = controlnet_images[0].shape[-2:]
            else:
                assert False

        # 5. Preprocess reference image
        logger.debug("Preprocessing reference image")
        if ref_image is not None:
            if isinstance(ref_image, PIL.Image.Image):
                ref_image = self.image_processor.preprocess(
                    ref_image, height=height, width=width
                )
            ref_image = self.norm_image_tensor(ref_image)
        # 6. Prepare timesteps
        logger.debug("Preparing timesteps")
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        logger.debug("Preparing latent variables")
        num_channels_latents = self.unet.config.in_channels
        if image is not None:
            if isinstance(image, PIL.Image.Image):
                image = self.image_processor.preprocess(
                    image, height=height, width=width
                )
            if isinstance(image, torch.Tensor):
                image = self.norm_image_tensor(image)
                input_latents = self.image_to_latents(
                    image,
                    batch_size * num_images_per_prompt,
                    self.unet.dtype,
                    device,
                    generator,
                    False,  # it will duplicate the latents after this step
                )

        logger.debug("Preparing latents")
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.unet.dtype,
            device,
            generator,
            latents,
            image,
            latent_timestep,
            is_strength_max=strength == 1.0,
            return_noise=True,
            return_image_latents=return_image_latents,
        )
        if return_image_latents:
            input_latents, noise, image_latents = latents_outputs
        else:
            input_latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        if mask_image is not None:
            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width
            )
            init_image = image
            init_image = init_image.to(dtype=torch.float32)
            if masked_image_latents is None:
                masked_image = init_image * (mask_condition < 0.5)
            else:
                masked_image = masked_image_latents
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

        # 8. Prepare reference latent variables
        if ref_image is not None:
            ref_image_latents = self.image_to_latents(
                ref_image,
                batch_size * num_images_per_prompt,
                self.unet.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ref_mask_dict, out_mask_dict = self.get_ref_mask_dicts(
            ref_image_mask,
            height,
            width,
            num_images_per_prompt,
            write_mask,
            bool_mask,
            device,
            batch_size,
        )
        ref_data = ReferenceData(
            ref_image=ref_image,
            ref_image_mask=ref_image_mask,
            style_fidelity=style_fidelity,
            attention_auto_machine_weight=attention_auto_machine_weight,
            gn_auto_machine_weight=gn_auto_machine_weight,
            ref_mask_dict=ref_mask_dict,
            out_mask_dict=out_mask_dict,
        )
        if reference_attn:
            self.unet = ReferenceOnlyUNet2DConditionModel.from_unet(
                self.unet,
                ref_data,
                reference_attn,
                reference_adain,
            )
        else:
            self.unet = ReferenceOnlyUNet2DConditionModel.revert_unet(
                self.unet
            )  # 9. Modify self attention and group norm
        if ref_image is not None:
            self.unet.ref_data.MODE = "write"
            self.unet.ref_data.uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt
                    + [0] * batch_size * num_images_per_prompt
                )
                .type_as(ref_image_latents)
                .bool()
            )

        if self.controlnet:
            # Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )

        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ref only part
                if reference_attn:
                    self.unet.ref_data.progress = i / num_inference_steps

                if ref_image is not None:
                    single_shape = (1,) + ref_image_latents.shape[1:]
                    single_noise = randn_tensor(
                        single_shape,
                        generator=generator,
                        device=device,
                        dtype=ref_image_latents.dtype,
                    )
                    noise_for_ref = single_noise.repeat_interleave(
                        ref_image_latents.shape[0], dim=0
                    )
                    ref_xt = self.scheduler.add_noise(
                        ref_image_latents,
                        noise_for_ref,
                        t.reshape(
                            1,
                        ),
                    )
                    # ref_xt = self.scheduler.scale_model_input(ref_xt, t)

                    self.unet.ref_data.MODE = "write"
                    self.unet(
                        ref_xt,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )
                    self.unet.ref_data.MODE = "read"

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([input_latents] * 2)
                    if do_classifier_free_guidance
                    else input_latents
                )

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = input_latents
                    control_model_input = self.scheduler.scale_model_input(
                        control_model_input, t
                    )
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                # calculate final conditioning_scale
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [
                        c * s
                        for c, s in zip(
                            controlnet_conditioning_scale, controlnet_keep[i]
                        )
                    ]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                assert isinstance(
                    self.controlnet, (ControlNetModel, MultiControlNetModel)
                )
                if n_controlnet_unit != 0:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_images,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [
                            torch.cat([torch.zeros_like(d), d])
                            for d in down_block_res_samples
                        ]
                        mid_block_res_sample = torch.cat(
                            [
                                torch.zeros_like(mid_block_res_sample),
                                mid_block_res_sample,
                            ]
                        )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )["sample"]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                input_latents = self.scheduler.step(
                    noise_pred, t, input_latents, **extra_step_kwargs
                )["prev_sample"]
                if num_channels_unet == 4 and (
                    mask_image is not None or masked_image_latents is not None
                ):
                    init_latents_proper = image_latents
                    if do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    input_latents = (
                        1 - init_mask
                    ) * init_latents_proper + init_mask * input_latents
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, input_latents)
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type != "latent":
            result_imgs = self.vae.decode(
                input_latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            result_imgs, has_nsfw_concept = self.run_safety_checker(
                result_imgs, device, prompt_embeds.dtype
            )
        else:
            result_imgs = input_latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * result_imgs.shape[0]
        else:
            if isinstance(has_nsfw_concept, list):
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            else:
                do_denormalize = [not has_nsfw_concept]
        # nan to zero
        result_imgs = torch.nan_to_num(result_imgs, nan=0.0, posinf=0.0, neginf=0.0)
        result_imgs = self.image_processor.postprocess(
            result_imgs, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (result_imgs, has_nsfw_concept)
        img_out = self.get_img_from_latents(latents=input_latents)

        modules = torch_dfs(self.unet)
        # 卸载 ref only hack
        for module in modules:

            if getattr(module, "_original_inner_forward", None) is not None:
                # unregister the attn forward hook
                module.forward = module._original_inner_forward

            if getattr(module, "original_forward", None) is not None:
                # unregister the adain forward hook
                module.forward = module.original_forward

        return StableDiffusionPipelineOutput(
            images=img_out, nsfw_content_detected=has_nsfw_concept
        )

    def get_ref_mask_dicts(
        self,
        ref_image_mask,
        height,
        width,
        num_images_per_prompt,
        write_mask,
        bool_mask,
        device,
        batch_size,
    ):
        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor
        ref_mask_dict = {}
        out_mask_dict = {}
        for i in range(4):
            w = latent_width >> i
            h = latent_height >> i

            resize_wh = (w, h)
            if ref_image_mask:
                # resize ref_iamge_mask
                tmp_mt_key = mask_images_to_float_tensor(
                    [ref_image_mask],
                    resize_wh=resize_wh,
                ).to(device=device, dtype=self.unet.dtype)

                mt_key = (
                    tmp_mt_key.flatten() > 0.5 if bool_mask else tmp_mt_key.flatten()
                )
                ref_mask_dict[mt_key.shape[-1]] = mt_key.repeat(
                    batch_size * num_images_per_prompt, 1
                )

            if write_mask:
                tmp_mt_query = mask_images_to_float_tensor(
                    [write_mask],
                    resize_wh=resize_wh,
                ).to(device=device, dtype=self.unet.dtype)
                if bool_mask:
                    mt_query = tmp_mt_query.flatten(1) > 0.5
                else:
                    mt_query = tmp_mt_query.flatten(1)
                out_mask_dict[mt_query.shape[-1]] = mt_query.repeat(
                    batch_size * num_images_per_prompt, 1
                )

        return ref_mask_dict, out_mask_dict

    def norm_image_tensor(self, ref_image):
        # 如果 image 维度为 3，说明没有 batch 维度
        if len(ref_image.shape) == 3:
            # 增加 batch 维度
            ref_image = ref_image.unsqueeze(0)
        if ref_image.shape[3] == 3:
            # 转换成 channel 在前的形式
            ref_image = ref_image.permute(0, 3, 1, 2)
        if ref_image.min() >= 0:
            # 数值为 0 ~ 1
            # 数值规范到 -1 ~ 1
            ref_image = (ref_image * 2 - 1).clamp(-1, 1)
        return ref_image

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            Tuple[ControlNetModel],
            MultiControlNetModel,
        ] = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        if controlnet is None:
            controlnet = []
        pipe_class_name = self.__class__.__name__
        self.set_progress_bar_config(
            desc=f"Running {pipe_class_name}...",
            unit_scale=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.unet: UNet2DConditionModel
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def image_to_latents(
        self,
        image,
        batch_size,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(
                    generator=generator[i]
                )
                for i in range(batch_size)
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(
                generator=generator
            )
        image_latents = self.vae.config.scaling_factor * image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if image_latents.shape[0] < batch_size:
            if batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1
            )

        image_latents = (
            torch.cat([image_latents] * 2)
            if do_classifier_free_guidance
            else image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        image_latents = image_latents.to(device=device, dtype=dtype)
        return image_latents

    # def decode_latents(self, latents: torch.Tensor):
    #     return self.get_img_from_latents(latents=latents)

    def get_img_from_latents(self, latents: torch.Tensor):
        # scale and decode the image latents with vae
        if len(latents.shape) == 3:
            latents = latents[None]
        norm_latents = latents
        dec_tensor = self.vae.decode(
            norm_latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        dec_images = self.image_processor.postprocess(
            dec_tensor, output_type="np", do_denormalize=[True] * dec_tensor.shape[0]
        )
        dec_image_zero = dec_images
        dec_image_zero = np.nan_to_num(dec_image_zero)
        image_out_np = np.clip(
            (dec_image_zero * 255.0).round().astype(int), a_min=0, a_max=255
        ).astype(np.uint8)
        return [PIL.Image.fromarray(img) for img in image_out_np]

    def encode_images_to_latents(
        self,
        imgs: List[PIL.Image.Image],
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        width, height = imgs[0].size
        image_tensor = _images_to_tensors(
            imgs=imgs, width=width, height=height, device=device, dtype=dtype
        )
        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            image_latent = torch.cat(
                [
                    self.vae.encode(image_tensor[i : i + 1]).latent_dist.sample(
                        generator=generator[i]
                    )
                    for i in range(image_tensor.shape[0])
                ],
                dim=0,
            )
        else:
            image_latent = self.vae.encode(image_tensor).latent_dist.sample(
                generator=generator
            )
        image_latent = self.vae.config.scaling_factor * image_latent

        return image_latent.to(device=device, dtype=dtype)

    def get_timesteps(self, num_inference_steps, strength: float):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if return_image_latents or (latents is None and not is_strength_max):
            # TODO: check it
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if is_strength_max:
                latents = noise * self.scheduler.init_noise_sigma
            else:
                latents = self.scheduler.add_noise(image_latents, noise, timestep)
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if is_strength_max:
                latents = noise * self.scheduler.init_noise_sigma
            else:
                latents = self.scheduler.add_noise(latents, noise, timestep)

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)
        return outputs

    def prepare_mask_latents(
        self,
        mask: torch.Tensor,
        masked_image: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(
                masked_image, generator=generator
            )

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if batch_size % mask.shape[0] != 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if batch_size % masked_image_latents.shape[0] != 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        image_latents = self.vae.config.scaling_factor * image_latents
        return image_latents


@dataclass
class ReferenceData:
    ref_image: Union[torch.FloatTensor, PIL.Image.Image] = None
    ref_image_mask: Union[torch.FloatTensor, PIL.Image.Image] = None
    MODE: str = "write"
    progress: float = 0.0
    uc_mask: torch.Tensor = None
    bool_mask: bool = True
    style_fidelity: float = 1.0
    do_classifier_free_guidance: bool = True
    attention_auto_machine_weight: float = 100.0
    gn_auto_machine_weight: float = 1.0
    ref_mask_dict: dict = None
    out_mask_dict: dict = None


class ReferenceOnlyUNet2DConditionModel(UNet2DConditionModel):
    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        ref_data: ReferenceData = ReferenceData(),
        reference_attn: bool = False,
        reference_adain: bool = False,
    ) -> "ReferenceOnlyUNet2DConditionModel":
        # 创建一个新的子类实例
        basic_transformer_idx = 0
        basic_transformer_blocks = []
        for module in torch_dfs(unet):
            if reference_attn:
                if isinstance(module, BasicTransformerBlock):
                    basic_transformer_blocks.append(module)
                    module.__class__ = BasicTransformerBlockReferenceOnly
                    module.ref_data = ref_data
                    module.bank = []
                    module.idx = basic_transformer_idx
                    basic_transformer_idx += 1
                elif reference_adain:
                    if isinstance(module, CrossAttnDownBlock2D):
                        module.__class__ = CrossAttnDownBlock2DReferenceOnly
                        module.ref_data = ref_data
                        module.bank = []
                    if isinstance(module, DownBlock2D):
                        module.__class__ = DownBlock2DReferenceOnly
                    if isinstance(module, UNetMidBlock2DCrossAttn):
                        module.__class__ = UNetMidBlock2DCrossAttnReferenceOnly
                    if isinstance(module, UpBlock2D):
                        module.__class__ = UpBlock2DReferenceOnly
                    if isinstance(module, CrossAttnUpBlock2D):
                        module.__class__ = CrossAttnUpBlock2DReferenceOnly
                    module.ref_data = ref_data
                    unet.mid_block.gn_weight = 0
                    down_blocks = unet.down_blocks
                    module.mean_bank = []
                    module.var_bank = []
                    for w, module in enumerate(down_blocks):
                        module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                        module.gn_weight *= 2

                    up_blocks = unet.up_blocks
                    for w, module in enumerate(up_blocks):
                        module.gn_weight = float(w) / float(len(up_blocks))
                        module.gn_weight *= 2

        # 计算 attn_weight
        basic_transformer_blocks = sorted(
            basic_transformer_blocks, key=lambda x: -x.norm1.normalized_shape[0]
        )

        for i, module in enumerate(basic_transformer_blocks):
            module.attn_weight = float(i) / float(len(basic_transformer_blocks))
        unet.__class__ = cls
        unet.ref_data = ref_data
        return unet

    @classmethod
    def revert_unet(
        cls, unet: "ReferenceOnlyUNet2DConditionModel"
    ) -> UNet2DConditionModel:
        unet.__class__ = UNet2DConditionModel
        for module in torch_dfs(unet):
            if isinstance(module, BasicTransformerBlockReferenceOnly):
                module.__class__ = BasicTransformerBlock
            if isinstance(module, CrossAttnDownBlock2DReferenceOnly):
                module.__class__ = CrossAttnDownBlock2D
            if isinstance(module, DownBlock2DReferenceOnly):
                module.__class__ = DownBlock2D
            if isinstance(module, UNetMidBlock2DCrossAttnReferenceOnly):
                module.__class__ = UNetMidBlock2DCrossAttn
            if isinstance(module, UpBlock2DReferenceOnly):
                module.__class__ = UpBlock2D
            if isinstance(module, CrossAttnUpBlock2DReferenceOnly):
                module.__class__ = CrossAttnUpBlock2D
        return unet


class BasicTransformerBlockReferenceOnly(BasicTransformerBlock):

    @classmethod
    def from_module(
        cls, module: BasicTransformerBlock
    ) -> "BasicTransformerBlockReferenceOnly":
        module.__class__ = cls
        return module

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: torch.LongTensor | None = None,
        _: Dict[str, torch.Tensor] | None = None,
    ) -> torch.FloatTensor:
        assert isinstance(self.idx, int)
        ref_data = self.ref_data
        assert isinstance(ref_data, ReferenceData)
        bank = self.bank
        assert isinstance(bank, list)

        uc_mask = ref_data.uc_mask
        bool_mask = ref_data.bool_mask
        ref_mask_dict = ref_data.ref_mask_dict

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.norm1(
                hidden_states,
                timestep,
                class_labels,
                hidden_dtype=hidden_states.dtype,
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # 1. Self-Attention
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )

        if self.only_cross_attention:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=(
                    encoder_hidden_states if self.only_cross_attention else None
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        else:
            if ref_data.MODE == "write":
                bank.append(norm_hidden_states.detach().clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if self.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            if ref_data.MODE == "read":
                style_fidelity = ref_data.style_fidelity
                attention_auto_machine_weight = ref_data.attention_auto_machine_weight
                do_classifier_free_guidance = ref_data.do_classifier_free_guidance
                if attention_auto_machine_weight > self.attn_weight:
                    attn_output_uc = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat(
                            [norm_hidden_states] + self.bank, dim=1
                        ),
                        # attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    attn_output_c = attn_output_uc.clone()
                    if do_classifier_free_guidance and style_fidelity > 0:
                        attn_output_c[uc_mask] = self.attn1(
                            norm_hidden_states[uc_mask],
                            encoder_hidden_states=norm_hidden_states[uc_mask],
                            **cross_attention_kwargs,
                        )
                    attn_output = (
                        style_fidelity * attn_output_c
                        + (1.0 - style_fidelity) * attn_output_uc
                    )
                    bank.clear()
                else:
                    # 原始的自注意力（无 reference only
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=(
                            encoder_hidden_states if self.only_cross_attention else None
                        ),
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = attn_output + hidden_states

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class CrossAttnDownBlock2DReferenceOnly(CrossAttnDownBlock2D):
    @classmethod
    def from_module(
        cls, module: CrossAttnDownBlock2D
    ) -> "CrossAttnDownBlock2DReferenceOnly":
        module.__class__ = cls
        return module

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        cross_attention_kwargs: Dict[str, Any] | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        additional_residuals: torch.FloatTensor | None = None,
    ) -> Tuple[torch.FloatTensor | Tuple[torch.FloatTensor]]:
        MODE = self.ref_data.MODE
        gn_auto_machine_weight = self.ref_data.gn_auto_machine_weight
        do_classifier_free_guidance = self.ref_data.do_classifier_free_guidance
        style_fidelity = self.ref_data.style_fidelity
        uc_mask = self.ref_data.uc_mask

        eps = 1e-6
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            if MODE == "write" and gn_auto_machine_weight >= self.gn_weight:
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                self.mean_bank.append([mean])
                self.var_bank.append([var])
            if MODE == "read" and (len(self.mean_bank) > 0 and len(self.var_bank) > 0):
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                hidden_states_c = hidden_states_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    hidden_states_c[uc_mask] = hidden_states[uc_mask]
                hidden_states = (
                    style_fidelity * hidden_states_c
                    + (1.0 - style_fidelity) * hidden_states_uc
                )

            output_states = output_states + (hidden_states,)

        if MODE == "read":
            self.mean_bank = []
            self.var_bank = []

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2DReferenceOnly(DownBlock2D):
    @classmethod
    def from_module(cls, module: DownBlock2D):
        instance = cls()
        instance.__dict__.update(module.__dict__)
        return instance

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor | Tuple[torch.FloatTensor]]:

        MODE = self.ref_data.MODE
        gn_auto_machine_weight = self.ref_data.gn_auto_machine_weight
        do_classifier_free_guidance = self.ref_data.do_classifier_free_guidance
        style_fidelity = self.ref_data.style_fidelity
        uc_mask = self.ref_data.uc_mask

        eps = 1e-6
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            if MODE == "write" and gn_auto_machine_weight >= self.gn_weight:
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                self.mean_bank.append([mean])
                self.var_bank.append([var])
            if MODE == "read" and (len(self.mean_bank) > 0 and len(self.var_bank) > 0):
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                hidden_states_c = hidden_states_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    hidden_states_c[uc_mask] = hidden_states[uc_mask]
                hidden_states = (
                    style_fidelity * hidden_states_c
                    + (1.0 - style_fidelity) * hidden_states_uc
                )

            output_states = output_states + (hidden_states,)

        if MODE == "read":
            self.mean_bank = []
            self.var_bank = []

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttnReferenceOnly(UNetMidBlock2DCrossAttn):
    @classmethod
    def from_module(cls, module: UNetMidBlock2DCrossAttn):
        instance = cls()
        instance.__dict__.update(module.__dict__)
        return instance

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        cross_attention_kwargs: Dict[str, Any] | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        return super().forward(
            hidden_states,
            temb,
            encoder_hidden_states,
            attention_mask,
            cross_attention_kwargs,
            encoder_attention_mask,
        )

    def forward(self, *args, **kwargs):
        MODE = self.ref_data.MODE
        gn_auto_machine_weight = self.ref_data.gn_auto_machine_weight
        do_classifier_free_guidance = self.ref_data.do_classifier_free_guidance
        style_fidelity = self.ref_data.style_fidelity
        uc_mask = self.ref_data.uc_mask
        eps = 1e-6
        x = super().forward(*args, **kwargs)
        if MODE == "write" and gn_auto_machine_weight >= self.gn_weight:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
            self.mean_bank.append(mean)
            self.var_bank.append(var)
        if MODE == "read":
            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                var_acc = sum(self.var_bank) / float(len(self.var_bank))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                x_uc = (((x - mean) / std) * std_acc) + mean_acc
                x_c = x_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    x_c[uc_mask] = x[uc_mask]
                x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
            self.mean_bank = []
            self.var_bank = []
        return x


class UpBlock2DReferenceOnly(UpBlock2D):
    @classmethod
    def from_module(cls, module: UpBlock2D):
        instance = cls()
        instance.__dict__.update(module.__dict__)
        return instance

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        MODE = self.ref_data.MODE
        gn_auto_machine_weight = self.ref_data.gn_auto_machine_weight
        do_classifier_free_guidance = self.ref_data.do_classifier_free_guidance
        style_fidelity = self.ref_data.style_fidelity
        uc_mask = self.ref_data.uc_mask

        eps = 1e-6
        for i, resnet in enumerate(self.resnets):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

            if MODE == "write" and gn_auto_machine_weight >= self.gn_weight:
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                self.mean_bank.append([mean])
                self.var_bank.append([var])
            if MODE == "read" and (len(self.mean_bank) > 0 and len(self.var_bank) > 0):
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                hidden_states_c = hidden_states_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    hidden_states_c[uc_mask] = hidden_states[uc_mask]
                hidden_states = (
                    style_fidelity * hidden_states_c
                    + (1.0 - style_fidelity) * hidden_states_uc
                )

        if MODE == "read":
            self.mean_bank = []
            self.var_bank = []

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class CrossAttnUpBlock2DReferenceOnly(CrossAttnUpBlock2D):
    @classmethod
    def from_module(cls, module: CrossAttnUpBlock2D):
        instance = cls()
        instance.__dict__.update(module.__dict__)
        return instance

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor],
        temb: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        cross_attention_kwargs: Dict[str, Any] | None = None,
        upsample_size: int | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:

        MODE = self.ref_data.MODE
        gn_auto_machine_weight = self.ref_data.gn_auto_machine_weight
        do_classifier_free_guidance = self.ref_data.do_classifier_free_guidance
        style_fidelity = self.ref_data.style_fidelity
        uc_mask = self.ref_data.uc_mask

        eps = 1e-6
        # TODO(Patrick, William) - attention mask is not used
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            if MODE == "write" and gn_auto_machine_weight >= self.gn_weight:
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                self.mean_bank.append([mean])
                self.var_bank.append([var])
            if MODE == "read" and (len(self.mean_bank) > 0 and len(self.var_bank) > 0):
                var, mean = torch.var_mean(
                    hidden_states, dim=(2, 3), keepdim=True, correction=0
                )
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                hidden_states_c = hidden_states_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    hidden_states_c[uc_mask] = hidden_states[uc_mask]
                hidden_states = (
                    style_fidelity * hidden_states_c
                    + (1.0 - style_fidelity) * hidden_states_uc
                )

        if MODE == "read":
            self.mean_bank = []
            self.var_bank = []

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
