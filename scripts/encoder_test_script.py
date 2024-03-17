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
vae = config_mng.get_autoencoder_worker()
quant = config_mng.get_quant_worker()
compressor = config_mng.get_compress_worker()


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


def encoder_pipeline(input_image):
    global vae
    global quant
    global compressor
    global traced_model

    # img = cv2.imread(input_image)
    img = input_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    # img = kill_artifacts(img, delta=25)
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img)
    img = img.cuda()

    img = img.to(torch.float16)
    img = img / 255.0
    current_shape = img.shape
    img = img.reshape(1, *current_shape)

    # Что-то с устройством можно сюда

    with torch.no_grad():
        if vae:
            latent_img, _ = vae.encode_work(img)
        else:
            latent_img = img

        if quant:
            (latent_img, quant_params), _ = quant.quant_work(latent_img)
        latent_img, _ = compressor.compress_work(latent_img)

        return latent_img


def main():
    global config_mng

    # Тут просто для теста, нужно заменить на нормальное получение картинки
    # base = "1"
    # input_image = "dependence/img_test/{}.png".format(base)
    input_video = "materials/dataset/air/2/7.mp4"

    config_parse = config_mng.config
    socket_settings = config_parse["SocketSettings"]
    statistics_settings = config_parse["StatisticsSettings"]
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
