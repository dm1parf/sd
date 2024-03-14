import argparse
import configparser
import socket
import cv2
import torch
import os
import zlib
import struct
import time
import sys
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from outer_models.util import instantiate_from_config
from outer_models.network_swinir import SwinIR
from outer_models.prediction.model.models import Model as Predictor, DMVFN, VPvI
from outer_models.realesrgan import RealESRGANer, RealESRGANModel
from basicsr.archs.rrdbnet_arch import RRDBNet

# import torch_tensorrt
import pytorch_lightning as pl


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

    if os.path.isfile("vq-f16_optimized.ts"):
        traced_model = torch.jit.load("vq-f16_optimized.ts").cuda()
    else:
        config = OmegaConf.load(f"{config}")
        model = load_model_from_config(config, f"{ckpt}")
        # model = model.type(torch.float16).cuda()

        model = model.to(torch.float16).cuda()  # float32
        model.forward = model.decode
        # model = model.to_torchscript()
        model._trainer = pl.Trainer()
        inp = [torch.randn(1, 8, 32, 32, dtype=torch.float16, device='cuda')]
        traced_model = torch.jit.trace(model, inp)

        print("Компиляция декодера завершена!")
        torch.jit.save(traced_model, "vq-f16_optimized.ts")

    return traced_model


def load_sr(pth, upscale=1, noise_aspect=0.5, height=1080, width=1920):
    """Загрузка модели SR."""

    backend_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    dni_base = noise_aspect
    dni_weight = [dni_base, 1-dni_base]
    model = RealESRGANer(scale=upscale, model_path=pth, dni_weight=dni_weight, tile=0, tile_pad=10, pre_pad=0,
                         model=backend_model, half=False)

    return model


def decoder_pipeline(model, latent_img):
    """Пайплайн декодирования"""

    latent_img = deflated_decompress(latent_img)
    latent_img = latent_img.float().cuda()
    latent_img = dequantinize_sigmoid(latent_img)

    # latent_img = latent_img.type(torch.float16)
    # output_img = model.decode(latent_img)
    latent_img = latent_img.half().cuda()  # float
    output_img = model.forward(latent_img)

    # output_img = output_img.to(dtype=torch.float16)

    return output_img


def sr_pipeline(model, latent_img: torch.Tensor, height: int, width: int):
    """Пайплайн сверхразрешения"""

    # new_img = torchvision.transforms.functional.resize(latent_img, [height, width],
    #                                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    # new_img = model(new_img)
    new_img = cv2.resize(latent_img, [width, height])
    new_img = model.enhance(new_img, outscale=2)[0]

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
        # default="outer_models/pth/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
        # default="outer_models/pth/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
        # default="outer_models/pth/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth",
        default="outer_models/pth/RealESRGAN_x2plus.pth",
        help="Путь к весам к модели сверхразрешения (SR)",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
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
    parser.add_argument(
        "--noise_aspect",
        type=int,
        default=0.75,
        help="the seed (for reproducible sampling)",
    )

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    opt = parser.parse_args()
    seed_everything(opt.seed)

    decoder_model = load_decoder(opt.config, opt.ckpt)


    config_parse = configparser.ConfigParser()
    config_parse.read(os.path.join("outer_models", "test_scripts", "decoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    screen_settings = config_parse["ScreenSettings"]
    pipeline_settings = config_parse["PipelineSettings"]
    internal_stream_settings = config_parse["InternalStreamSettings"]
    enable_sr = bool(int(pipeline_settings["enable_sr"]))
    enable_predictor = bool(int(pipeline_settings["enable_predictor"]))

    height = int(screen_settings["height"])
    width = int(screen_settings["width"])
    if enable_predictor:
        pred_model = DMVFN(load_path=opt.pred_pth).cuda()  # TODO: predictor loader
        pred_module = Predictor(pred_model)
    if enable_sr:
        sr_model = load_sr(opt.sr_pth, upscale=opt.upscale, noise_aspect=opt.noise_aspect, height=height, width=width)

    internal_stream_mode = bool(int(internal_stream_settings["stream_used"]))
    if internal_stream_mode:
        internal_stream_sock_data = (internal_stream_settings["host"], int(internal_stream_settings["port"]))
        internal_socket = socket.socket()
        internal_socket.connect(internal_stream_sock_data)

    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    new_socket.bind((socket_host, socket_port))
    #new_socket.listen(1)

    cv2.destroyAllWindows()
    print("--- Ожидаем данные с кодировщика ---")
    # connection, address = new_socket.accept()
    len_all = 65536
    len_seq = 4
    len_defer = 4
    len_ender = 10
    cont_ender = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'

    i = 0
    predict_img = None
    while True:
        buffer = {}
        max_seq = 0
        address = None
        while True:
            starter = len_seq

            datagram_payload, new_address = new_socket.recvfrom(len_all)
            if not address:
                address = new_address
            else:
                if new_address != address:
                    continue
            seq = struct.unpack('I', datagram_payload[:len_seq])[0]
            if seq > max_seq:
                max_seq = seq
            if seq == 0:
                starter += len_defer
                image_len = struct.unpack('I', datagram_payload[len_seq:starter])[0]

            buffer[seq] = datagram_payload[starter:-len_ender]
            ender_bytes = datagram_payload[-len_ender:]
            if ender_bytes != cont_ender:
                break

        latent_image = b''
        for seq in range(max_seq+1):
            latent_image += buffer[seq]

        new_socket.sendto(b'\x01', address)

        if len(latent_image) != image_len:
            print("Неправильная длина изображения:", len(latent_image), "!=", image_len)
            continue

        a_time = time.time()
        new_img: torch.tensor = decoder_pipeline(decoder_model, latent_image)
        b_time = time.time()

        if internal_stream_mode and (predict_img is not None):
            img_bytes = predict_img.tobytes()
            len_struct = struct.pack("I", len(img_bytes))
            internal_socket.send(len_struct + img_bytes)
        elif predict_img is not None:
            cv2.imshow("===", predict_img)
            cv2.waitKey(1)

        # Дальше можно в CV2.
        new_img = new_img * 255.0
        new_img = new_img.to(torch.uint8)

        new_img = new_img.squeeze(0)
        new_img = new_img.permute(1, 2, 0)
        new_img = new_img.detach()
        new_img = new_img.cpu()
        new_img = new_img.numpy()
        if new_img.any():
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

            c_time = time.time()
            if enable_sr:
                new_img = sr_pipeline(sr_model, new_img, height // opt.upscale, width // opt.upscale)
            else:
                new_img = cv2.resize(new_img, [width, height])
            d_time = time.time()

            if internal_stream_mode:
                img_bytes = new_img.tobytes()
                len_struct = struct.pack("I", len(img_bytes))
                internal_socket.send(len_struct + img_bytes)
            else:
                cv2.imshow("===", new_img)
                cv2.waitKey(1)

            e_time = time.time()
            if enable_predictor:
                new_img = cv2.resize(new_img, (1024 * 2, 512 * 2))
                predict_img = pred_module.predict([new_img, new_img], 1)
                predict_img = cv2.resize(predict_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            f_time = time.time()

            all_time = []
            decoder_pipeline_time = round(b_time - a_time, 3)
            all_time.append(decoder_pipeline_time)
            decoder_pipeline_fps = round(1 / decoder_pipeline_time, 3)
            between_decoder_sr_time = round(c_time - b_time, 3)
            all_time.append(between_decoder_sr_time)
            if between_decoder_sr_time != 0:
                between_decoder_sr_fps = round(1 / between_decoder_sr_time, 3)
            else:
                between_decoder_sr_fps = "oo"
            if enable_sr:
                sr_pipeline_time = round(d_time - c_time, 3)
                all_time.append(sr_pipeline_time)
                sr_pipeline_fps = round(1 / sr_pipeline_time, 3)
            if enable_predictor:
                predict_time = round(f_time - e_time, 3)
                all_time.append(predict_time)
                predict_fps = round(1 / predict_time, 3)
            draw_time = round(e_time - d_time, 3)
            if draw_time != 0:
                draw_fps = round(1 / draw_time, 3)
            else:
                draw_fps = "oo"
            total_time = round(f_time - a_time, 3)
            total_fps = round(1 / total_time, 3)
            nominal_time = round(sum(all_time), 3)
            nominal_fps = round(1 / nominal_time, 3)

            print(f"--- Время выполнения: {i} ---")
            print("- Декодер:", decoder_pipeline_time, "с / FPS:", decoder_pipeline_fps)
            print("- Зазор 1:", between_decoder_sr_time, "с / FPS:", between_decoder_sr_fps)
            if enable_sr:
                print("- SR:", sr_pipeline_time, "с / FPS", sr_pipeline_fps)
            if enable_predictor:
                print("- Предикт:", predict_time, "с / FPS", predict_fps)
            print("- Отрисовка/отправка:", draw_time, "с / FPS:", draw_fps)
            print("- Итого (всего):", total_time, "с / FPS", total_fps)
            print("- Итого (номинально):", nominal_time, "с / FPS", nominal_fps)
            print()

        i += 1


if __name__ == "__main__":
    main()
