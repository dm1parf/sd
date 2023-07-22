import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
import logging
import cv2
from constants.constant import DENOISE_STEPS
logger = logging.getLogger('main')

class BonchSDInpPipeline(StableDiffusionInpaintPipeline):

    @torch.no_grad()
    def bonch_encode(self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image, mask_image, height, width, return_image=True
        )

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float16,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else: 
            latents, noise = latents_outputs
            image_latents = None

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            torch.float16,
            device,
            generator,
            do_classifier_free_guidance,
        )
        init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
        init_image = self._encode_vae_image(init_image, generator=generator)
        encoded_vae_image = init_image
        non_zero_mask_values = 0
        for value in mask.flatten():
            if value != 0:
                # logger.debug(f'value: {value}')
                non_zero_mask_values += 1
        logger.debug(f'non zero mask values: {non_zero_mask_values}')
        print(f'latents made')

        return ((latents, masked_image_latents, mask))
    
    @torch.no_grad()
    def bonch_decode(self, bonch_tuple = None):
        if bonch_tuple is not None:
            (bonch_tuple_2) = bonch_tuple
            latents, masked_image_latents, mask = bonch_tuple_2
        # print('________________')
        # print(f'latents {latents}')
        # print(f'masked_image_latents {masked_image_latents}')
        # print(f'mask {mask}')
        print('________________')
        print(f'latents {latents.shape}')
        print(f'masked_image_latents {masked_image_latents.shape}')
        print(f'mask {mask.shape}')
        print('________________')
        device = self._execution_device
        num_inference_steps = DENOISE_STEPS
        guidance_scale = float(7.5)
        do_classifier_free_guidance = guidance_scale > 1.0

        eta = 0.0
        generator = None
        output_type = "pil"
        return_dict = True
        callback = None
        callback_steps = 1
        cross_attention_kwargs = None
        image_latents = None
        num_images_per_prompt = 1
        negative_prompt = None
        negative_prompt_embeds = None
        prompt_embeds = None
        prompt_embeds = self._encode_prompt(
            '',
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=None,
        )
        
        # for element in bonch_tuple:
        #     print(element)
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )
        print(f'num_channels_unet {num_channels_unet}')
        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # print(f'num_inference_steps {num_inference_steps}')
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=1.0, device=device
        )
        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ---------------Для получения латентных картинок расскомментить эти 2 строчки и запустить, 
                # для разных выводов поменять mask на latents или masked_image_latents
                # image = mask
                # break
                # --------------
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # if num_channels_unet == 4:
                #     init_latents_proper = image_latents[:1]
                #     init_mask = mask[:1]

                #     if i < len(timesteps) - 1:
                #         noise_timestep = timesteps[i + 1]
                #         init_latents_proper = self.scheduler.add_noise(
                #             init_latents_proper, noise, torch.tensor([noise_timestep])
                #         )

                #     latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # print(image)
            
            # cv2.imwrite('test.jpg', image)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        # if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        # print(has_nsfw_concept)
        if not return_dict:
            return (image, [False])

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=[False])
