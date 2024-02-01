import argparse
import os
import configparser
import socket
import cv2
import numpy as np
from omegaconf import OmegaConf
import zlib
import torch
from pytorch_lightning import seed_everything
from util import instantiate_from_config
import torch_tensorrt
import struct
# import sys

# sys.path.append('/NIR/Docker-Volumes/Storage/Users/Parfenov/nir')

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


def encoder_pipeline(model, input_image):
    
    # img = cv2.imread(input_image)
    img = input_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
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

def get_frame_rtsp(rtsp_uri):
    cap = cv2.VideoCapture(rtsp_uri)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            continue
        
        yield frame

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="/Storage/nir/outer_models/config/vq-f16.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/Storage/nir/outer_models/ckpt/vq-f16.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    # inputs = [
    #     torch_tensorrt.Input(
    #         min_shape=[1, 1, 32, 32],
    #         opt_shape=[1, 1, 32, 32],
    #         max_shape=[1, 1, 32, 32],
    #         dtype=torch.half,
    #     )
    # ]
    # enabled_precisions = {torch.float, torch.half}

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.cuda()
    model = model.type(torch.float16)

    # Тут просто для теста, нужно заменить на нормальное получение картинки
    base = "1"
    input_image = "img_test/{}.png".format(base)

    config_parse = configparser.ConfigParser()
    config_parse.read(os.path.join("test_scripts", "encoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket()
    new_socket.connect((socket_host, socket_port))
    video_settings = config_parse["VideoSettings"]
    rtsp_uri = video_settings['rtsp_uri']
    import time
    # for i in range(1000):
    for i, frame in enumerate(get_frame_rtsp(rtsp_uri)):
        a = time.time()
        input_image = frame
        latent_img = encoder_pipeline(model, input_image)
        b = time.time()

        image_length = len(latent_img)
        img_bytes = struct.pack('I', image_length)

        print(i, "---", round(b - a, 5), "с")

        new_socket.send(img_bytes + latent_img)

        try:
            print('wait for new byte')
            new_byte = new_socket.recv(1)
            print('new byte got!')
            if not (new_byte == b'\x01'):
                break
        except ConnectionResetError:
            print("Сервер разорвал соединение.")
            break

    # TODO: В ЗАВИСИМОСТИ ОТ ЛОГИКИ ВВОДА!
    new_socket.close()


if __name__ == "__main__":
    main()
