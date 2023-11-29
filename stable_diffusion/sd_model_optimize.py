import inspect
import time
import numpy as np
import torch
from PIL import Image
from stable_diffusion.constant import TORCH_DEVICE
from common.logging_sd import configure_logger
from stable_diffusion.Diffusion.utilities import TRT_LOGGER
from stable_diffusion.Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline
import tensorrt as trt

logger = configure_logger(__name__)

class SdModelOptimize(StableDiffusionPipeline):
    def __init__(self) -> None:
        super().__init__( scheduler="PNDM",
            denoising_steps=6,
            output_dir=r"output",
            version="2.1",
            hf_token=None,
            verbose=False,
            nvtx_profile=False, # разобраться, что делает
            max_batch_size=16,
            stages=['vae_encoder', 'clip', 'unet', 'vae'])
        
        self.loadEngines(r"stable_diffusion/Diffusion/engine", 
                 r"stable_diffusion/Diffusion/onnx", 
                 17,
                 opt_batch_size=1, 
                 opt_image_height=512, 
                 opt_image_width=512,
                 force_export=False, 
                 force_optimize=False, 
                 force_build=False,
                 static_batch=False, 
                 static_shape=True,
                 enable_refit=False, 
                 enable_all_tactics=False,
                 timing_cache=None, 
                 onnx_refit_dir=None)
        self.loadResources(512, 512, 1, None)
        self.prompt = [""]
        self.negative_prompt = [""]
        self.strength = 0.04

    def to_latents(self, img: Image):
        """
        Конвертирует изображение в латентное пространство с помощью модели VAE.
        :param img -- RGB изображение в формате PIL.Image.
        :return: latents -- латентный вектор, полученный из модели VAE, тип - torch.Tensor.
        """
        
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            
            np_img = (np.array(img).astype(np.float16) / 255.0) * 2.0 - 1.0
            np_img = np_img[None].transpose(0, 3, 1, 2)
            torch_img = torch.from_numpy(np_img).contiguous()
            
            # start = time.time()
            
            latents = self.encode_image(torch_img)
            
            torch.cuda.synchronize()
            
            # print(f"Encode = {time.time() - start}")

        return latents
        
    def denoise(self, latents):
        """
        Очищает зашумленные latents с помощью Unet.

        :param latents: Тензор, содержащий шумные значения latents
        :type latents: torch.Tensor

        :return: Тензор, содержащий очищенные значения latents
        :rtype: torch.Tensor
        """
        latents = latents * 0.18215
        batch_size = len(self.prompt)
        assert len(self.prompt) == len(self.negative_prompt)
        
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(self.denoising_steps, self.strength)
            latent_timestep = timesteps[:1].repeat(batch_size)

            # CLIP text encoder
            text_embeddings = self.encode_prompt(self.prompt, self.negative_prompt)

            # Add noise to latents using timesteps
            noise = torch.randn(latents.shape, generator=self.generator, device=self.device, dtype=torch.float32)
            latents = self.scheduler.add_noise(latents, noise, t_start, latent_timestep)

            # start = time.time()
            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, timesteps=timesteps, step_offset=t_start)
            
            torch.cuda.synchronize()
            
            # print(f"Unet = {time.time() - start}")
            
        return latents / 0.18215
    
    def to_img(self, latents):
        """
        Преобразует вектор latents в изображение.

        :param latents: `torch.Tensor` с shape (batch_size, latent_size).
        :return: `PIL.Image` объект.
        """
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            
            # start = time.time()
            
            torch_img = self.decode_latent(latents)
                        
            torch.cuda.synchronize()
            # print(f"Decode = {time.time() - start}")
            
            torch_img = (torch_img / 2 + 0.5).clamp(0, 1)
            np_img = torch_img.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
            np_img = (np_img * 255.0).astype(np.uint8)
            img = Image.fromarray(np_img)

        
        return img

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

    
    