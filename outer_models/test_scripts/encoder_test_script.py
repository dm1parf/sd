import argparse
import os
import configparser
import signal
import socket
import cv2
import numpy as np
import zlib
import torch
import struct
import csv
import time
import sys
from omegaconf import OmegaConf
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from pytorch_lightning import seed_everything
from outer_models.util import instantiate_from_config


# Сигмоидальное квантование
def quantinize_sigmoid(latent_img: torch.Tensor, hardcode=False):
    if hardcode:  # Так не нужно передавать, но в зависимости от картинки хуже
        miner = 2.41
    else:
        miner = latent_img.min().item()

    new_img = torch.clone(latent_img)
    new_img -= miner
    new_img = 1 / (1 + torch.exp(-new_img))
    new_max = torch.max(new_img).item()

    if hardcode:
        scaler = 256.585
    else:
        scaler = 255 / new_max
    new_img *= scaler
    new_img = torch.round(new_img)
    new_img = new_img.to(torch.uint8)
    new_img = new_img.clamp(0, 255)

    quant_params = [miner, scaler]
    return new_img, quant_params


# Сжатие с DEFLATED
def deflated_method(latent_img: torch.tensor):
    """Сжатие латента с помощью DEFLATE."""

    numpy_img = latent_img.numpy()
    byters = numpy_img.tobytes()

    compresser = zlib.compressobj(level=9, method=zlib.DEFLATED)
    byter = byters
    new_min = compresser.compress(byter)
    new_min += compresser.flush()
    # print("Сжатый латент:", len(new_min) / 1024, "Кб")

    return new_min


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


def kill_artifacts(img, delta=15):
    # 5 более-менее подавляет, но так себе.
    # 10 тоже достаточно хорош, но бывают немалые артефакты
    # 15 -- эмпирически лучший.
    # 25-35 очень хороши в подавлении артефактов, но заметно искажают цвета в артефактогенных местах

    low = delta
    high = 255 - delta

    low_array = np.array([low, low, low])
    high_array = np.array([high, high, high])
    low_mask = np.sum(img < low, axis=2) == 3
    high_mask = np.sum(img > high, axis=2) == 3

    img[low_mask] = low_array
    img[high_mask] = high_array

    return img


def encoder_pipeline(model, input_image):
    # img = cv2.imread(input_image)
    img = input_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    """
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = -1/256 * np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, -476, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]])
    img = cv2.filter2D(img, -1, kernel)
    """
    img = kill_artifacts(img, delta=25)
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img)
    img = img.cuda()

    img = img.to(torch.float16)
    img = img / 255.0
    current_shape = img.shape
    img = img.reshape(1, *current_shape)

    # Что-то с устройством можно сюда

    model.eval()
    with torch.no_grad():
        output = model.encode(img)
        latent_img, loss, info = output

        # Чтобы не передавать данные -- hardcode
        # Но можно потом и передавать
        latent_img, quant_params = quantinize_sigmoid(latent_img, hardcode=True)

        # print("Размер латента:", latent_img.shape)

        latent_img = latent_img.cpu()  # Иначе не будет работать здесь
        latent_img = deflated_method(latent_img)

        return latent_img


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
    model = model.cuda()
    model = model.type(torch.float16)

    # Тут просто для теста, нужно заменить на нормальное получение картинки
    # base = "1"
    # input_image = "outer_models/img_test/{}.png".format(base)
    input_video = "outer_models/img_test/7.mp4"

    config_parse = configparser.ConfigParser()
    config_parse.read(os.path.join("outer_models", "test_scripts", "encoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    statistics_settings = config_parse["StatisticsSettings"]
    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket()
    new_socket.connect((socket_host, socket_port))

    # Просто сделать поле пустым, когда статистику собирать не надо
    stat_filename = statistics_settings["file"]
    if stat_filename:
        stat_file = open(stat_filename, 'w', newline='')
        csv_stat = csv.writer(stat_file)
        csv_stat.writerow(["id", "bin_size"])

    def urgent_close(*_):
        nonlocal new_socket
        nonlocal stat_file
        new_socket.close()
        stat_file.close()
    signal.signal(signal.SIGINT, urgent_close)

    i = 0
    cap = cv2.VideoCapture(input_video)

    while True:
        for _ in range(2):
            ret, frame = cap.read()
            if not ret:
                break

        a = time.time()
        latent_img = encoder_pipeline(model, frame)
        b = time.time()

        image_length = len(latent_img)
        if stat_filename:
            csv_stat.writerow([i, image_length])
            stat_file.flush()
        img_bytes = struct.pack('I', image_length)

        print(i, "---", round(b - a, 5), "с")

        new_socket.send(img_bytes + latent_img)

        try:
            new_byte = new_socket.recv(1)
            if not (new_byte == b'\x01'):
                break
        except ConnectionResetError:
            print("Сервер разорвал соединение.")
            break

        i += 1

    urgent_close()


if __name__ == "__main__":
    main()
