import cv2
import numpy as np
import torch
from PIL import Image

from constants.constant import DEVICE
from stable_diffusion.constant import TORCH_DEVICE
from stable_diffusion_inp.bonch_sd_pipe import BonchSDInpPipeline


def quantize(latents):
    """
    Квантуем тензор с latents до значений в диапазоне от 0 до 255.

    :param latents: Тензор, содержащий латентные векторы.
    :type latents: torch.Tensor

    :return: ndarray, содержащий квантованные значения в диапазоне от 0 до 255.
    :rtype: np.ndarray
    """
    quantized_latents = (latents / (255 * 0.18215) + 0.5).clamp(0, 1)
    quantized = quantized_latents.cpu().permute(0, 2, 3, 1).detach().numpy()
    quantized = (quantized * 255.0 + 0.5).astype(np.uint8)
    return quantized


def unquantize(quantized):
    """
    Преобразует массив, закодированный целочисленными значениями в несжатый тензор типа float32.

    :param quantized: Массив значений, закодированных целочисленными значениями (uint8).
    :type quantized: np.ndarray

    :return: Несжатый тензор типа float32, содержащий преобразованные значения.
    :rtype: torch.Tensor
    """
    unquantized = quantized.astype(np.float32) / 255.0
    if unquantized.shape[0] == 2:
        unquantized = unquantized.transpose(0, 3, 1, 2)
    else:
        unquantized = unquantized[0][None].transpose(0, 3, 1, 2)
    unquantized_latents = (unquantized - 0.5) * (255 * 0.18215)
    unquantized_latents = torch.from_numpy(unquantized_latents)
    return unquantized_latents.to(TORCH_DEVICE)


class SdInpCompressor:
    def __init__(self):
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        self.pipe = BonchSDInpPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            # revision="fp16",
            torch_dtype=torch.float32,
        )

        self.pipe = self.pipe.to(DEVICE)

        self.prompt = ''

        self.denoise_steps = 5

    def quantize_img(self, img):
        img = prepare_img(img)

        mask_img = create_img_mask(img)
        img.save("img.png")
        mask_img.save("mask.png")

        latents, masked_image_latents, mask = self.pipe.bonch_encode(prompt=self.prompt, image=img, mask_image=mask_img)

        q_latents = quantize(latents)
        q_masked_image_latents = quantize(masked_image_latents)
        q_mask = quantize(mask)
        return q_latents, q_masked_image_latents, q_mask

    def uncompress(self, encoded_tuple: tuple):
        q_latents, q_masked_image_latents, q_mask = encoded_tuple
        latents = unquantize(q_latents)
        masked_image_latents = unquantize(q_masked_image_latents)
        mask = unquantize(q_mask)
        result_img = self.pipe.bonch_decode(bonch_tuple=(latents, masked_image_latents, mask)).images[0]
        return result_img


def create_img_mask(img):
    gray_image = np.full((512, 512), 0, dtype=np.uint8)
    # print(img)
    img = np.array(img)
    image_for_mask = img.copy()

    imgray = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_image = cv2.drawContours(gray_image, contours, -1, (255, 255, 255), 1)

    mask_image = cv2.resize(mask_image, (512, 512))

    mask_im_pil = Image.fromarray(mask_image)
    return mask_im_pil


def prepare_img(img):
    image = cv2.resize(img, (512, 512))
    img_pil = Image.fromarray(image)
    return img_pil
