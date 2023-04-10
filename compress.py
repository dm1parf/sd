import time
from copy import copy

import libimagequant as liq
import numpy as np
from PIL import Image

from helpers import denoise, quantize, to_img, to_latents, unquantize
from metrics import pirson, ssim
from utils import rescaled_pil_img_and_write_using_cv2, write_metrics, write_txt

maxStapeDenoise = 3


def quantize_img (latents, img_name="default", dir_name="test", is_save=True):
    quantized = quantize(latents)

    quantized_img = Image.fromarray(quantized)

    if is_save:
        quantized_img.save(f"data/output/{dir_name}/2_{img_name}.png", lossless=True, quality=100)

    # further quantize to palette. Use libimagequant for Dithering
    attr = liq.Attr()
    attr.speed = 1
    attr.max_colors = 256
    input_image = attr.create_rgba(quantized.flatten('C').tobytes(),
                                   quantized_img.width,
                                   quantized_img.height,
                                   0)

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
    latents = unquantize(latents)

    if is_save:
        palettized_img = to_img(latents)
        palettized_img.save(f"data/output/{dir_name}/3_{img_name}.png")

    return latents


def ns_run (img, img_name="default", dir_name="test", is_quantize=True, is_save=True, save_metrics=True,
            save_rescaled_out=False):
    start_time = time.time()
    # Display VAE roundtrip image
    latents = to_latents(img)

    latents_time = time.time()

    if is_save:
        img_from_latents = to_img(latents)
        img_from_latents.save(f"data/output/{dir_name}/0_{img_name}.png")

    save = copy(latents)
    if is_save:
        img_from_latents = to_img(latents)
        img_from_latents.save(f"data/output/{dir_name}/1_{img_name}.png")
        write_txt(f"data/output/{dir_name}/latents_{img_name}.txt", save.cpu().detach().numpy())

    quantize_time = None
    if is_quantize:
        latents = quantize_img(latents, img_name, dir_name, is_save)
        quantize_time = time.time()

    denoise_step_time = 0
    for stapeDenoise in range(maxStapeDenoise):
        latents = denoise(latents)

        if denoise_step_time == 0:
            denoise_step_time = time.time()

    denoise_time = time.time()

    denoised_img = to_img(latents)

    finish_time = time.time()

    if is_save:
        denoised_img.save(f"data/output/{dir_name}/4_{img_name}.png")

    if save_rescaled_out:
        rescaled_pil_img_and_write_using_cv2(denoised_img, f"data/output/{dir_name}/4_{img_name}_rescaled.png")

    if save_metrics:  # TODO: it is bad work; it  using (512;512) shape and not rescaled img;
        img1 = (np.array(img))
        img2 = (np.array(denoised_img.resize(img.shape)))
        ssim_data = 1 - ssim.ssim(img1, img2)

        pirson_data = pirson.cor_pirson(img1, img2)

        latents_time_data = latents_time - start_time

        checkpoint = 0
        if quantize_time is not None:
            quantize_time_data = quantize_time - latents_time
            checkpoint = quantize_time
        else:
            quantize_time_data = "None"
            checkpoint = latents_time

        denoise_step_time_data = denoise_step_time - checkpoint
        denoise_time_data = denoise_time - checkpoint
        finish_time_data = finish_time - start_time

        data_str = f"ssim_data={ssim_data}\n" \
                   f"pirson_data={pirson_data}\n" \
                   f"latents_time_data={latents_time_data}\n" \
                   f"quantize_time_data={quantize_time_data}\n" \
                   f"denoise_step_time_data={denoise_step_time_data}\n" \
                   f"denoise_time_data={denoise_time_data}\n" \
                   f"finish_time_data={finish_time_data}\n"

        write_metrics(f"data/output/{dir_name}/metrics_{img_name}.txt", data_str)


