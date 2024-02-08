"""make variations of input image"""
"""Модифицированная """

import socket
import cv2
import argparse
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
import os
import torchvision
import torchvision.transforms.functional as tvfunc
from pytorch_lightning import seed_everything
# import pillow_avif
# from pillow_heif import register_heif_opener, register_avif_opener
import math
from skimage.metrics import structural_similarity

from outer_models.util import instantiate_from_config


# Здесь большая часть из img2img.
# Оставил на всякий случай

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


# Данную функцию использую, хотя можно и без неё
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def quantinize_power(latent_img: torch.Tensor):
    maxer = latent_img.max().item()
    miner = latent_img.min().item()
    aller = maxer - miner
    scaler = math.log(255, aller)

    new_img = torch.clone(latent_img)
    new_img = (new_img - miner) ** scaler
    new_img = new_img.to(torch.uint8)
    new_img = new_img.clamp(0, 255)

    quant_params = [miner, scaler]
    return new_img, quant_params


def quantinize_sigmoid(latent_img: torch.Tensor):
    miner = latent_img.min().item()

    new_img = torch.clone(latent_img)
    new_img -= miner
    new_img = 1 / (1 + torch.exp(-new_img))
    new_max = torch.max(new_img).item()
    scaler = 255 / new_max
    new_img *= scaler
    new_img = torch.round(new_img)
    new_img = new_img.to(torch.uint8)
    new_img = new_img.clamp(0, 255)

    quant_params = [miner, scaler]
    return new_img, quant_params


def quantinize_linear(latent_img: torch.Tensor):
    maxer = latent_img.max().item()
    miner = latent_img.min().item()
    aller = maxer - miner
    scaler = 255 / aller

    new_img = torch.clone(latent_img)
    new_img = (new_img - miner) * scaler
    new_img = new_img.to(torch.uint8)
    new_img = new_img.clamp(0, 255)

    quant_params = [miner, scaler]
    return new_img, quant_params


def dequantinize_power(quant: torch.Tensor, quant_params):
    miner, scaler = quant_params

    new_img = torch.clone(quant)
    new_img = new_img.to(torch.float32)
    new_img = (new_img ** (1 / scaler)) + miner

    return new_img


def dequantinize_sigmoid(quant: torch.Tensor, quant_params):
    miner, scaler = quant_params

    new_img = torch.clone(quant)
    new_img = new_img.to(torch.float32)
    new_img /= scaler
    new_img = -torch.log((1 / (new_img)) - 1)
    new_img += miner

    return new_img


def dequantinize_linear(quant: torch.Tensor, quant_params):
    miner, scaler = quant_params

    new_img = torch.clone(quant)
    new_img = new_img.to(torch.float32)
    new_img = (new_img / scaler) + miner

    return new_img


def deflated_method(latent_img: torch.tensor, *_):
    this_shape = list(latent_img.shape)

    numpy_img = latent_img.numpy()
    byters = numpy_img.tobytes()

    import zlib
    import lzma
    import gzip
    compresser = zlib.compressobj(level=9, method=zlib.DEFLATED)
    byter = byters
    new_min = compresser.compress(byter)
    new_min += compresser.flush()
    # new_min = lzma.compress(byter)
    print("Сжатый латент:", len(new_min) / 1024, "Кб")

    decompressor = zlib.decompressobj()
    byters = decompressor.decompress(new_min)
    byters += decompressor.flush()

    latent_img = torch.frombuffer(byters, dtype=torch.uint8)
    latent_img = latent_img.reshape(*this_shape)

    return latent_img


def jpeg_method(latent_img: torch.tensor, latent_image: str, *_):
    """Мой новый модифицированный метод."""
    # 8 каналов!

    this_shape = list(latent_img.shape)
    neo_shape = this_shape.copy()
    neo_shape.pop(1)
    if neo_shape[1] > neo_shape[2]:
        neo_shape[2] *= 2
        neo_shape[1] *= 4
    else:
        neo_shape[1] *= 2
        neo_shape[2] *= 4
    latent_img = latent_img.reshape(*neo_shape)  # 1, 64, 128
    pillow_img: PIL.Image.Image = tvfunc.to_pil_image(latent_img, mode="L")  # RGBA
    # 60 -- linear, power
    # 70 --
    pillow_img.save(latent_image, "JPEG", optimize=True, quality=30)  # +++
    file_size = os.stat(latent_image).st_size / 1024
    print("Итоговый файл:", file_size, "Кб")
    latent_img = torchvision.io.read_image(latent_image, mode=torchvision.io.ImageReadMode.UNCHANGED)
    latent_img = latent_img.reshape(*this_shape)  # 1, 8, 32, 32

    return latent_img


def jpeg8_method(latent_img: torch.tensor, latent_image: str, *_):
    """Мой новый модифицированный метод."""
    # 4 канала!

    this_shape = list(latent_img.shape)
    neo_shape = this_shape.copy()
    neo_shape.pop(1)
    neo_shape[1] *= 2
    neo_shape[2] *= 2
    latent_img = latent_img.reshape(*neo_shape)  # 1, 64, 128
    pillow_img: PIL.Image.Image = tvfunc.to_pil_image(latent_img, mode="L")  # RGBA
    pillow_img.save(latent_image, "JPEG", optimize=True, quality=30)  # +++
    file_size = os.stat(latent_image).st_size / 1024
    print("Итоговый файл:", file_size, "Кб")
    latent_img = torchvision.io.read_image(latent_image, mode=torchvision.io.ImageReadMode.UNCHANGED)
    latent_img = latent_img.reshape(*this_shape)  # 1, 4, 32, 32

    return latent_img


def avif_method(latent_img: torch.tensor, latent_image: str, *_):
    """Ещё метод."""
    # 8 каналов!

    register_heif_opener()
    register_avif_opener()

    latent_image = latent_image.replace('l.png', 'l.avif')  # j2k, webp, avif, heic

    this_shape = list(latent_img.shape)
    neo_shape = this_shape.copy()
    neo_shape.pop(1)
    if neo_shape[1] > neo_shape[2]:
        neo_shape[2] *= 2
        neo_shape[1] *= 4
    else:
        neo_shape[1] *= 2
        neo_shape[2] *= 4
    latent_img = latent_img.reshape(*neo_shape)  # 1, 64, 128
    pillow_img: PIL.Image.Image = tvfunc.to_pil_image(latent_img, mode="L")  # RGBA
    # pillow_img.save(latent_image, "AVIF", optimize=True, quality=60)  # +++
    pillow_img.save(latent_image, optimize=True, quality=60)
    # pillow_img.save(latent_image, optimize=True, quality_mode='rates', quality_layers=[3])
    file_size = os.stat(latent_image).st_size / 1024
    print("Итоговый файл:", file_size, "Кб")
    pillow_img = PIL.Image.open(latent_image).convert(mode='L')
    latent_img = tvfunc.pil_to_tensor(pillow_img)
    latent_img = latent_img.reshape(*this_shape)  # 1, 8, 32, 32

    return latent_img


def magic_method(latent_img: torch.tensor, latent_image: str, *_):
    import imageio as iio

    # 8 каналов!

    iio.plugins.freeimage.download()

    latent_image = latent_image.replace('l.png', 'l.hdp')

    this_shape = list(latent_img.shape)
    neo_shape = this_shape.copy()
    neo_shape.pop(1)
    if neo_shape[1] > neo_shape[2]:
        neo_shape[2] *= 2
        neo_shape[1] *= 4
    else:
        neo_shape[1] *= 2
        neo_shape[2] *= 4
    latent_img = latent_img.reshape(*neo_shape)  # 1, 64, 128
    latent_img = latent_img.numpy()
    latent_img = np.moveaxis(latent_img, 0, 2)
    iio.imwrite(latent_image, latent_img, format="JPEG-XR")
    file_size = os.stat(latent_image).st_size / 1024
    print("Итоговый файл:", file_size, "Кб")
    latent_img = iio.imread(latent_image, format='JPEG-XR')
    latent_img = torch.from_numpy(latent_img)
    latent_img = latent_img.reshape(*this_shape)  # 1, 8, 32, 32

    return latent_img


# ==============

def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse

def ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image1, image2, data_range=image2.max() - image2.min())

    return score


def psnr_metric(image1, image2):
    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def print_metrics(source, dest):
    fromer = cv2.imread(source)
    toer = cv2.imread(dest)

    psnr = psnr_metric(fromer, toer)
    ssim = ssim_metric(fromer, toer)

    print("PSNR:", psnr)
    print("SSIM:", ssim)


def pipeline(model, model_type, test_image, latent_image, end_image,
             quant_mode, dequant_mode, compression_mode):
    # Старый вариант загрузки картинки через load_img

    # img = load_img(test_image)

    # Новый, лучший вариант загрузки картинки без альфы

    img = cv2.imread(test_image)
    print("Начальный размер:", img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_shape = list(img.shape)[:-1][::-1]
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    # Image.fromarray(img).show()
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img)

    # img = torchvision.io.read_image(test_image, mode=torchvision.io.ImageReadMode.RGB)
    # resizer = torchvision.transforms.Resize((512, 512))

    # torchvision.transforms.functional.to_pil_image(img).show()

    # img = resizer(img)

    img = img.to(torch.float32)
    # img.apply_(lambda x: x / 255.0)  # "Охота на комаров с дробовиком"
    img = img / 255.0
    current_shape = img.shape
    img = img.reshape(1, *current_shape)  # Добавление размерности батча -- 1 (можно так много картинок сложить)
    # print(img)  # Так, для тестов
    # img = load_img(test_image)
    # current_shape = list(img.shape)[1:]

    # Тут для себя, так как какая-то несовместимость у меня с CUDA у тех моделей
    # Если что, можно просто img = img.cuda(), а model в загрузке

    model = model.cpu()
    img = img.cpu()
    # model = model.cuda(); img = img.cuda()

    # Для тестов. Размерность тензора изначальной картинки

    # print(img.shape)

    # Без model.eval и torch.no_grad не стоит использовать модели

    model.eval()
    with (torch.no_grad()):
        output = model.encode(img)
        if model_type == "vq":  # Тут не распределение, а сразу картинка
            latent_img, loss, info = output
        elif model_type == "kl":  # Тут распределение
            latent_img = output.sample()

        # А здесь квантование и деквантование
        # linear, power, sigmoid
        latent_img, quant_params = quant_mode(latent_img)

        print(quant_params)

        torchvision.io.write_png(latent_img.reshape(1, 128, 64), "3__t.png")

        print("Размер латента:", latent_img.shape)

        # Здесь использовать метод после кванта (лучший jpeg_method)
        # jpeg_method, deflated_method, jpeg8_method, avif_method
        latent_img = compression_mode(latent_img, latent_image)

        latent_img = dequant_mode(latent_img, quant_params)

        new_img = model.decode(latent_img)

        # Размерность выходной картинки

        print("Конечный размер:", new_img.shape)

        new_img = new_img * 255.0

        new_img = new_img.to(torch.uint8)
        current_shape = list(new_img.shape)[1:]
        new_img = new_img.reshape(*current_shape)
        # torchvision.io.write_png(new_img, end_image, compression_level=6)

        # pillow_img: PIL.Image.Image = tvfunc.to_pil_image(new_img, mode="RGB")
        # pillow_img.save(jpeg_image, "JPEG", optimize=True, quality=5)  #+++

        new_img = new_img.numpy()
        new_img = np.moveaxis(new_img, 0, 2)
        new_img = cv2.resize(new_img, start_shape)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(end_image, new_img)

        print("END!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./outer_models/config/vq-f16.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./outer_models/ckpt/vq-f16.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    # А вот сей аргумент добавляю
    parser.add_argument(
        "-t", "--type",
        type=str,
        dest="type",
        help="Type of model",
        choices=["vq", "kl"],
        default="vq"
    )

    # Здесь по большей части как раньше

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Далее удалено и полностью переписано

    model_type = opt.type

    # quantinize_sigmoid quantinize_power quantinize_linear
    quant_mode = quantinize_sigmoid
    # dequantinize_sigmoid dequantinize_power dequantinize_linear
    dequant_mode = dequantinize_sigmoid
    # jpeg_method, deflated_method, jpeg8_method, avif_method
    compression_mode = deflated_method

    base = "1"
    test_image = r"./outer_models/img_test/{}.png".format(base)
    # Можно если что сохранить промежуточный с убранным альфа-каналом
    # pre_image = r"test_magic\apple2.png"
    latent_image = r"./outer_models/img_test{}l.png".format(base)
    end_image = r"./outer_models/img_test{}_.png".format(base)
    jpeg_image = r"./outer_models/img_test{}.jpg".format(base)
    
    pipeline(model, model_type, test_image, latent_image, end_image,
             quant_mode, dequant_mode, compression_mode)
    """

    folder = r"test_magic\micro_dataset"
    files = os.listdir(folder)
    for file in files:
        if ("_.jpg" in file) or ("l.jpg" in file):
            continue
        print("\n===", file)
        test_image = os.path.join(folder, file)
        latent_image = os.path.join(folder, file.replace(".jpg", "l.jpg"))
        end_image = os.path.join(folder, file.replace(".jpg", "_.jpg"))
        pipeline(model, model_type, test_image, latent_image, end_image, quant_mode, dequant_mode, compression_mode)
        print_metrics(test_image, end_image)
    """


if __name__ == "__main__":
    main()
