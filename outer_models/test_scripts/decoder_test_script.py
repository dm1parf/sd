import argparse
import configparser
import socket
import cv2
import numpy as np
import torch
import os
import signal
from pytorch_lightning import seed_everything
from outer_models.util import instantiate_from_config
from omegaconf import OmegaConf
import zlib
import struct
import time


# Сигмоидальное квантование
def dequantinize_sigmoid(quant: torch.Tensor, quant_params=None):
    if not quant_params:  # Если не передавать отдельно
        quant_params = [2.41, 256.585]
    miner, scaler = quant_params

    new_img = torch.clone(quant)
    new_img = new_img.to(torch.float16)
    new_img /= scaler
    new_img = -torch.log((1 / new_img) - 1)
    new_img += miner

    return new_img


# Сжатие с DEFLATED
def deflated_decompress(latent_img: bytes):
    """Расжатие латента с помощью DEFLATE."""

    decompressor = zlib.decompressobj()
    byters = decompressor.decompress(latent_img)
    byters += decompressor.flush()

    latent_img = torch.frombuffer(byters, dtype=torch.uint8)
    latent_img = latent_img.reshape(1, 8, 32, 32)

    return latent_img


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

    model.eval()
    return model


def decoder_pipeline(model, latent_img):
    latent_img = deflated_decompress(latent_img)
    latent_img = latent_img.cuda()  # Для ускорения
    latent_img = dequantinize_sigmoid(latent_img)

    output_img = model.decode(latent_img)

    return output_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="outer_models/config/vq-f16.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outer_models/ckpt/vq-f16.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.type(torch.float16).cuda()

    config_parse = configparser.ConfigParser()
    config_parse.read(os.path.join("outer_models", "test_scripts", "decoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    screen_settings = config_parse["ScreenSettings"]
    height = int(screen_settings["height"])
    width = int(screen_settings["width"])

    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket()
    new_socket.bind((socket_host, socket_port))
    new_socket.listen(1)

    connection, address = new_socket.accept()
    len_defer = 4

    i = 0
    while True:
        # Некоторые аннотации для подсказок
        latent_image: bytes = connection.recv(len_defer)
        if not latent_image:  # Иногда читает 0 байт
            time.sleep(0.05)
            continue
        # Декодер работает дольше!
        image_len = struct.unpack('I', latent_image)[0]
        latent_image: bytes = connection.recv(image_len)

        a = time.time()
        new_img: torch.tensor = decoder_pipeline(model, latent_image)
        b = time.time()
        print("-----", b-a)

        # Дальше можно в CV2.

        new_img = new_img * 255.0

        new_img = new_img.to(torch.uint8)
        current_shape = list(new_img.shape)[1:]
        new_img = new_img.reshape(*current_shape)

        new_img = new_img.cpu()
        new_img = new_img.numpy()
        new_img = np.moveaxis(new_img, 0, 2)
        new_img = cv2.resize(new_img, (width, height))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        # Здесь отображайте как хотите
        # cv2.imshow("Выходная картинка", new_img)
        cv2.imwrite(f"out_img{i}.png", new_img)
        i += 1


if __name__ == "__main__":
    main()
