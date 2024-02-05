import argparse
import configparser
import socket
import cv2
import torch
import os
import zlib
import struct
import time
import torchvision.transforms.functional
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from outer_models.util import instantiate_from_config
from outer_models.network_swinir import SwinIR
from outer_models.prediction.model.models import Model as Predictor, DMVFN, VPvI


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


def load_decoder(config, ckpt):
    """Загрузка модели VAE."""

    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    model = model.type(torch.float16).cuda()
    return model


def load_sr(pth, upscale=1):
    """Загрузка модели SR."""


    model = SwinIR(upscale=upscale, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')

    """
    model = SwinIR(upscale=upscale, in_chans=3, img_size=126, window_size=7,
                   img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    """
    """
    model = SwinIR(upscale=upscale, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    """

    pretrained_model = torch.load(pth)
    model.load_state_dict(pretrained_model["params"])
    model = model.cuda()

    return model


def decoder_pipeline(model, latent_img):
    """Пайплайн декодирования"""

    latent_img = deflated_decompress(latent_img)
    latent_img = latent_img.cuda()  # Для ускорения
    latent_img = dequantinize_sigmoid(latent_img)

    output_img = model.decode(latent_img)

    return output_img


def sr_pipeline(model, latent_img: torch.Tensor, height: int, width: int):
    """Пайплайн сверхразрешения"""

    new_img = torchvision.transforms.functional.resize(latent_img, [height, width])
    new_img = model(new_img)

    return new_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="outer_models/config/vq-f16.yaml",
        help="Путь конфигурации построения автокодировщика (VAE)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outer_models/ckpt/vq-f16.ckpt",
        help="Путь к весам к модели вариационного автокодировщика (VAE)",
    )
    parser.add_argument(
        "--sr_pth",
        type=str,
        default="outer_models/pth/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
        # default="outer_models/pth/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
        # default="outer_models/pth/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth",
        help="Путь к весам к модели сверхразрешения (SR)",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="Показатель увеличения размера при сверхразрешении",
    )
    parser.add_argument(
        "--pred_pth",
        type=str,
        default="outer_models/prediction/pre_trained/dmvfn_city.pkl",
        help="Путь к весам к модели предсказания",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    decoder_model = load_decoder(opt.config, opt.ckpt)
    sr_model = load_sr(opt.sr_pth, upscale=opt.upscale)
    pred_model = DMVFN(load_path=opt.pred_pth).cuda()  # TODO: predictor loader
    pred_module = Predictor(pred_model)

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
        try:
            latent_image: bytes = connection.recv(len_defer)
            if not latent_image:
                connection, address = new_socket.accept()
                continue
            image_len = struct.unpack('I', latent_image)[0]
            latent_image: bytes = connection.recv(image_len)
            # connection.send(b'\x01') # ЕСЛИ РАЗНЫЕ КОМПЬЮТЕРЫ!

            a = time.time()
            new_img: torch.tensor = decoder_pipeline(decoder_model, latent_image)
            b = time.time()

            # Дальше можно в CV2.
            # torchvision.transforms.functional
            if len(new_img[new_img == 0]) != 3 * 512 * 512:
                new_img = new_img * 255.0
                new_img = new_img.to(torch.uint8)

                ### TODO: REMOVE!
                # new_img = new_img.cpu()
                # sr_model = sr_model.cpu()
                new_img = sr_pipeline(sr_model, new_img, height // opt.upscale, width // opt.upscale)

                new_img = new_img.squeeze(0)
                new_img = new_img.permute(1, 2, 0)
                new_img = new_img.detach()
                new_img = new_img.cpu()
                c = time.time()
                new_img = new_img.numpy()
                # new_img = cv2.resize(new_img, (width, height))
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                d = time.time()
                print(i, "---", round(d - a, 5), round(d - b, 5), round(d - c, 5))

                # Здесь отображайте как хотите
                # cv2.imwrite(f"out_img{i}.png", new_img)
                cv2.imshow("===", new_img)
                # cv2.waitKey(1)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()

                new_img = cv2.resize(new_img, (1024, 512))
                predict_img = pred_module.predict([new_img, new_img], 10)
                for i in predict_img:
                    i = cv2.resize(i, (width, height))
                    cv2.imshow("===", i)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()

            i += 1

            connection.send(b'\x01')  # Если один компьютер!
        except (ConnectionResetError, socket.error):
            connection, address = new_socket.accept()
            continue


if __name__ == "__main__":
    main()
