import libimagequant as liq
import numpy as np
from PIL import Image
import pickle

from common.logging_sd import configure_logger
from stable_diffusion.constant import MAXSTAPEDENOISE
from stable_diffusion.sd_model import SdModel

logger = configure_logger(__name__)


class SdCompressor:
    def __init__(self, platform):
        logger.debug(f"Initialization SdCompressor")
        self.sd = SdModel(platform)

    def quantize_img(self, img):
        """
        Квантуем исходное изображение до 256 цветов.

        :param latents: Массив значений, представляющий входное изображение типа np.ndarray
        :return: Массив значений, представляющий преобразованное изображение типа np.ndarray
        """
        logger.debug(f"get new image; quantize him")

        latents = self.sd.to_latents(img)
        quantized = self.sd.quantize(latents)
        bin_quantized = pickle.dumps(quantized, protocol=2)


        # logger.debug(
        #     f"quantize img to successful; get new img size ({input_image.width}, {input_image.height})")
        return bin_quantized

    def quantization_result(self, input_image):
        logger.debug(f"get new image; unquantize him")

        # further quantize to palette. Use libimagequant for Dithering
        attr = liq.Attr()
        attr.speed = 1
        attr.max_colors = 256

        quantization_result = input_image.quantize(attr)
        quantization_result.dithering_level = 1.0
        # Get the quantization result
        out_pixels = quantization_result.remap_image(input_image)
        out_palette = quantization_result.get_palette()
        np_indices = np.frombuffer(out_pixels, np.uint8)
        np_palette = np.array([c for color in out_palette for c in color], dtype=np.uint8)

        # Display VAE decoding of dithered 8-bit latents
        np_indices = np_indices.reshape((input_image.height, input_image.width))
        palettized_latent_img = Image.fromarray(np_indices, mode='P')
        palettized_latent_img.putpalette(np_palette, rawmode='RGBA')
        latents = np.array(palettized_latent_img.convert('RGBA'))
        latents = self.sd.unquantize(latents)

        logger.debug(f"unquantize successful; getting tensor {len(latents)}")
        return latents

    def compress(self, img: np.ndarray):
        """
        Сжимает входное изображение.

        :param img: Массив значений, представляющий входное изображение типа np.ndarray
        :return: Массив значений, представляющий сжатое изображение типа np.ndarray
        """
        latents = self.sd.to_latents(img)
        input_image = self.quantize_img(latents)

        logger.debug(f"successful compress img")

        return input_image

    def uncompress(self, bin_quantized):
        quantized = pickle.loads(bin_quantized)

        quantized_img = Image.fromarray(quantized)

        attr = liq.Attr()
        attr.speed = 1
        attr.max_colors = 256
        input_image = attr.create_rgba(quantized.flatten('C').tobytes(),
                                       quantized_img.width,
                                       quantized_img.height,
                                       0)

        latents = self.quantization_result(input_image)

        for stapeDenoise in range(MAXSTAPEDENOISE):
            latents = self.sd.denoise(latents)

        denoised_img = self.sd.to_img(latents)
        logger.debug(f"successful uncompress img")

        return denoised_img