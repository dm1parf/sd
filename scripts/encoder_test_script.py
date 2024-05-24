import os
import signal
import socket
import cv2
import numpy as np
import torch
import struct
import csv
import time
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager


config_path = os.path.join("scripts", "encoder_config.ini")
config_mng = ConfigManager(config_path)
as_ = config_mng.get_as_worker()
vae = config_mng.get_autoencoder_worker()
quant = config_mng.get_quant_worker()
compressor = config_mng.get_compress_worker()


def encoder_pipeline(input_image):
    global as_
    global vae
    global quant
    global compressor

    img = input_image
    img, _ = as_.prepare_work(img)

    with torch.no_grad():
        if vae:
            latent_img, _ = vae.encode_work(img)
        else:
            latent_img = img

        if quant:
            (latent_img, quant_params), _ = quant.quant_work(latent_img)
        # TODO: INSERT IN THE ARCHITECTURE!
        """
        magic_tensor = latent_img[0][0].cpu().numpy().tolist()
        array_str = ""
        for l in magic_tensor:
            array_str += "\t".join([str(i) for i in l])
            array_str += "\n"
        print(array_str)
        """
        latent_img, _ = compressor.compress_work(latent_img)

        return latent_img


def main():
    global config_mng

    # Тут просто для теста, нужно заменить на нормальное получение картинки
    # base = "1"
    # input_image = "dependence/img_test/{}.png".format(base)
    # input_video = "materials/dataset/air/2/7.mp4"

    config_parse = config_mng.config
    socket_settings = config_parse["SocketSettings"]
    statistics_settings = config_parse["StatisticsSettings"]
    source_settings = config_parse["SourceSettings"]
    input_video = source_settings["source"]
    socket_host = socket_settings["address"]
    socket_port = int(socket_settings["port"])
    max_payload = int(socket_settings["max_payload"]) - 14
    socket_address = (socket_host, socket_port)
    new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    timeout = int(socket_settings["timeout"])
    new_socket.settimeout(timeout)

    stat_filename = statistics_settings["file"]
    if stat_filename:
        stat_file = open(stat_filename, 'w', newline='')
        csv_stat = csv.writer(stat_file)
        csv_stat.writerow(["id", "bin_size"])

    def urgent_close(*_):
        nonlocal new_socket
        if stat_filename:
            nonlocal stat_file
            stat_file.close()
        new_socket.close()
    signal.signal(signal.SIGINT, urgent_close)

    i = 0
    cap = cv2.VideoCapture(input_video)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(input_video)
            ret, frame = cap.read()

        a = time.time()
        latent_img = encoder_pipeline(frame)
        b = time.time()

        image_length = len(latent_img)
        if stat_filename:
            if not stat_file.closed:
                csv_stat.writerow([i, image_length])
                stat_file.flush()
        img_bytes = struct.pack('I', image_length)

        print(i, "---", round(b - a, 5), "с; ", round(len(latent_img) / 1024, 2), "Кб ---")


        payload = img_bytes + latent_img
        seq = 0
        pointer = 0
        wait_flag = False
        if len(payload) > (max_payload*3):
            wait_flag = True
        while pointer < len(payload):
            seq_bytes = struct.pack('I', seq)
            seq += 1
            payload_fragment = seq_bytes + payload[pointer:pointer+max_payload]
            pointer += max_payload
            if pointer < len(payload):
                payload_fragment += b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'
            else:
                payload_fragment += b'\x10\x09\x08\x07\x06\x05\x04\x03\x02\x01'
            new_socket.sendto(payload_fragment, socket_address)
            if wait_flag:
                time.sleep(0.001)

        try:
            new_byte, _ = new_socket.recvfrom(1)
            if not (new_byte == b'\x01'):
                break
        except (ConnectionResetError, TimeoutError):
            continue
        finally:
            i += 1

    urgent_close()


if __name__ == "__main__":
    main()
