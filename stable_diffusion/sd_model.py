import inspect
import time
import numpy as np
import torch
from PIL import Image
from torch import autocast
from stable_diffusion.constant import TORCH_DEVICE
from stable_diffusion.download_ns import load_sd
from common.logging_sd import configure_logger

logger = configure_logger(__name__)


class SdModel:
    def __init__(self, platform):
        self.vae, \
            self.unet, \
            self.scheduler, \
            self.text_encoder, \
            self.tokenizer, \
            self.uncond_input, \
            self.uncond_embeddings = load_sd(platform)

    @torch.no_grad()
    def to_latents(img: Image):
        """
        Конвертирует изображение в латентное пространство с помощью модели VAE.
        :param img -- RGB изображение в формате PIL.Image.
        :return: latents -- латентный вектор, полученный из модели VAE, тип - torch.Tensor.
        """
        np_img = (np.array(img).astype(np.float16) / 255.0) * 2.0 - 1.0
        np_img = np_img[None].transpose(0, 3, 1, 2)
        torch_img = torch.from_numpy(np_img)
        with autocast("cpu"):
            generator = torch.Generator("cpu").manual_seed(0)
            latents = vae.encode(torch_img.to(vae.dtype).to(torch_device)).latent_dist.sample(generator=generator)
        return latents

    @torch.no_grad()
    def to_img(latents):
        """
        Преобразует вектор latents в изображение.
        :param latents: `torch.Tensor` с shape (batch_size, latent_size).
        :return: `PIL.Image` объект.
        """
        with autocast("cpu"):
            torch_img = vae.decode(latents.to(vae.dtype).to(torch_device)).sample
        torch_img = (torch_img / 2 + 0.5).clamp(0, 1)
        np_img = torch_img.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        np_img = (np_img * 255.0).astype(np.uint8)
        img = Image.fromarray(np_img)
        return img

    @torch.no_grad()
    def to_latents(self, img: Image):
        """
        Конвертирует изображение в латентное пространство с помощью модели VAE.
        :param img -- RGB изображение в формате PIL.Image.
        :return: latents -- латентный вектор, полученный из модели VAE, тип - torch.Tensor.
        """
        # start = time.time()
        
        np_img = (np.array(img).astype(np.float16) / 255.0) * 2.0 - 1.0
        np_img = np_img[None].transpose(0, 3, 1, 2)
        torch_img = torch.from_numpy(np_img)
        with autocast("cpu"):
            generator = torch.Generator("cpu").manual_seed(0)
            latents = self.vae.encode(torch_img.to(self.vae.dtype).to(TORCH_DEVICE)).latent_dist.sample(   
                generator=generator)
            
        
        # logger.info(f"STEP ENCODER = {time.time() - start} seconds")
        return latents

    @torch.no_grad()
    def to_img(self, latents):
        """
        Преобразует вектор latents в изображение.

        :param latents: `torch.Tensor` с shape (batch_size, latent_size).
        :return: `PIL.Image` объект.
        """
        # start = time.time()
        with autocast("cpu"):
            torch_img = self.vae.decode(latents.to(self.vae.dtype).to(TORCH_DEVICE)).sample
        torch_img = (torch_img / 2 + 0.5).clamp(0, 1)
        np_img = torch_img.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        np_img = (np_img * 255.0).astype(np.uint8)
        img = Image.fromarray(np_img)
        # logger.info(f"STEP DECODER = {time.time() - start} seconds")
        return img

    @torch.no_grad()
    def denoise(self, latents):
        """
        Очищает зашумленные latents с помощью Unet.

        :param latents: Тензор, содержащий шумные значения latents
        :type latents: torch.Tensor

        :return: Тензор, содержащий очищенные значения latents
        :rtype: torch.Tensor
        """
        latents = latents * 0.18215
        step_size = 15
        num_inference_steps = self.scheduler.config.get("num_train_timesteps", 1000) // step_size
        strength = 0.04
        self.scheduler.set_timesteps(num_inference_steps)
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=TORCH_DEVICE)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = 0.9
        latents = latents.to(self.unet.dtype).to(TORCH_DEVICE)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        
        # start = time.time()
        with autocast('cpu'):
            for i, t in enumerate(self.scheduler.timesteps[t_start:]):
                noise_pred = self.unet(latents, t, encoder_hidden_states=self.uncond_embeddings).sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        # logger.info(f"STEP UNET = {time.time() - start} seconds")
        # reset scheduler to free cached noise predictions
        self.scheduler.set_timesteps(1)
        return latents / 0.18215

    def quantize(self, latents):
        """
        Квантуем тензор с latents до значений в диапазоне от 0 до 255.

        :param latents: Тензор, содержащий латентные векторы.
        :type latents: torch.Tensor

        :return: ndarray, содержащий квантованные значения в диапазоне от 0 до 255.
        :rtype: np.ndarray
        """
        quantized_latents = (latents / (255 * 0.18215) + 0.5).clamp(0, 1)
        quantized = quantized_latents.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        quantized = (quantized * 255.0 + 0.5).astype(np.uint8)
        return quantized

    def unquantize(self, quantized):
        """
        Преобразует массив, закодированный целочисленными значениями в несжатый тензор типа float32.

        :param quantized: Массив значений, закодированных целочисленными значениями (uint8).
        :type quantized: np.ndarray

        :return: Несжатый тензор типа float32, содержащий преобразованные значения.
        :rtype: torch.Tensor
        """
        unquantized = quantized.astype(np.float32) / 255.0
        unquantized = unquantized[None].transpose(0, 3, 1, 2)
        unquantized_latents = (unquantized - 0.5) * (255 * 0.18215)
        unquantized_latents = torch.from_numpy(unquantized_latents)
        return unquantized_latents.to(TORCH_DEVICE)
