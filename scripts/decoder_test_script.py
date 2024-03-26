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
from omegaconf import OmegaConf
import pytorch_lightning as pl
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from dependence.util import instantiate_from_config
from dependence.realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from utils.config import ConfigManager


config_path = os.path.join("scripts", "decoder_config.ini")
config_mng = ConfigManager(config_path)
vae = config_mng.get_autoencoder_worker()
quant = config_mng.get_quant_worker()
compressor = config_mng.get_compress_worker()
predictor = config_mng.get_predictor_worker()
sr = config_mng.get_sr_worker()


def decoder_pipeline(latent_img):
    """Пайплайн декодирования"""

    global compressor
    global quant
    global vae
    global traced_model

    if quant:
        dest_type = torch.uint8
    else:
        if vae:
            dest_type = vae.nominal_type
        else:
            dest_type = torch.float16
    if vae:
        dest_shape = vae.z_shape
    else:
        dest_shape = [1, 3, 512, 512]
    latent_img, _ = compressor.decompress_work(latent_img, dest_shape, dest_type)
    if quant:
        latent_img, _ = quant.dequant_work(latent_img)
    if vae:
        output_img, _ = vae.decode_work(latent_img)
    else:
        output_img = latent_img

    return output_img


def main():
    global config_mng

    config_parse = config_mng.config
    config_parse.read(os.path.join("scripts", "decoder_config.ini"))
    socket_settings = config_parse["SocketSettings"]
    screen_settings = config_parse["ScreenSettings"]
    internal_stream_settings = config_parse["InternalStreamSettings"]

    height = int(screen_settings["height"])
    width = int(screen_settings["width"])

    internal_stream_mode = bool(int(internal_stream_settings["stream_used"]))
    if internal_stream_mode:
        internal_stream_sock_data = (internal_stream_settings["host"], int(internal_stream_settings["port"]))
        internal_socket = socket.socket()
        internal_socket.connect(internal_stream_sock_data)

    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    new_socket.bind((socket_host, socket_port))

    cv2.destroyAllWindows()
    print("--- Ожидаем данные с кодировщика ---")
    len_all = 65536
    len_seq = 4
    len_defer = 4
    len_ender = 10
    cont_ender = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'

    i = 0
    predict_img = None
    with torch.no_grad():
        while True:
            buffer = {}
            max_seq = 0
            address = None
            while True:
                starter = len_seq

                try:
                    datagram_payload, new_address = new_socket.recvfrom(len_all)
                except ConnectionResetError:
                    time.sleep(0.5)
                    continue
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
            new_img: torch.tensor = decoder_pipeline(latent_image)
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
                new_img, _ = sr.sr_work(new_img, dest_size=[width, height])
                d_time = time.time()

                if internal_stream_mode:
                    img_bytes = new_img.tobytes()
                    len_struct = struct.pack("I", len(img_bytes))
                    internal_socket.send(len_struct + img_bytes)
                else:
                    cv2.imshow("===", new_img)
                    cv2.waitKey(1)

                e_time = time.time()
                if predictor:
                    new_img = cv2.resize(new_img, (1280, 640))
                    predict_img = predictor.predict_work([new_img, new_img], 1)[0]
                    predict_img = cv2.resize(predict_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
                f_time = time.time()

                all_time = []
                decoder_pipeline_time = round(b_time - a_time, 3)
                all_time.append(decoder_pipeline_time)
                if decoder_pipeline_time != 0:
                    decoder_pipeline_fps = round(1 / decoder_pipeline_time, 3)
                else:
                    decoder_pipeline_fps = "oo"
                between_decoder_sr_time = round(c_time - b_time, 3)
                all_time.append(between_decoder_sr_time)
                if between_decoder_sr_time != 0:
                    between_decoder_sr_fps = round(1 / between_decoder_sr_time, 3)
                else:
                    between_decoder_sr_fps = "oo"
                sr_pipeline_time = round(d_time - c_time, 3)
                all_time.append(sr_pipeline_time)
                sr_pipeline_fps = round(1 / sr_pipeline_time, 3)
                if predictor:
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
                print("- SR:", sr_pipeline_time, "с / FPS", sr_pipeline_fps)
                if predictor:
                    print("- Предикт:", predict_time, "с / FPS", predict_fps)
                print("- Отрисовка/отправка:", draw_time, "с / FPS:", draw_fps)
                print("- Итого (всего):", total_time, "с / FPS", total_fps)
                print("- Итого (номинально):", nominal_time, "с / FPS", nominal_fps)
                print()
            else:
                print(f"--- Чёрный кадр: {i} --")

            i += 1


if __name__ == "__main__":
    main()
