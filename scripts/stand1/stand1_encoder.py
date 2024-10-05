import argparse
import os
import signal
import socket
import struct
import csv
import time
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import math
import cv2
from datetime import datetime
import numpy as np
from scripts.stand1.stand1_decoder import ConfigurationGuardian

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)

cfg_num = 14
hard_port = 6571
ip = "172.16.204.14"
fps = 5
seglen = 1000
video = "dataset_preparation/4x.mp4"

arguments = argparse.ArgumentParser(prog="Эмулятор кодера FPV CTVP",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-v", "--video", dest="video", type=str, default=video,
                       help="Номер конфигурации FPV CTVP")
arguments.add_argument("-c", "--cfg", dest="cfg", type=int, default=cfg_num,
                       help="Номер конфигурации FPV CTVP")
arguments.add_argument("-d", "--dest_ip", dest="ip", type=str, default=ip,
                       help="IP-адрес назначения")
arguments.add_argument("-f", "--fps", dest="fps", type=int, default=fps,
                       help="FPS кодера")
arguments.add_argument("-s", "--seglen", dest="seglen", type=int, default=seglen,
                       help="Длина сегментации бинарной последовательности")
args = arguments.parse_args()

video = args.video
cfg_num = args.cfg
ip = args.ip
fps = args.fps
seglen = args.seglen
lat_dir = "dataset_preparation/latent_dataset_cfg{}".format(cfg_num)
waiter = 1 / fps
inter_segment_waiter = 0.001

cfg_guard = ConfigurationGuardian()
neuro_codec = cfg_guard.get_configuration(cfg_num)

socket_address = (ip, hard_port)
new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# timeout = 5
# new_socket.settimeout(timeout)

stat_enable = True
if stat_enable:
    stat_filename = "stand1_encoder_stat.csv"
    stat_file = open(stat_filename, 'w', newline='')
    csv_stat = csv.writer(stat_file)
    csv_stat.writerow(["frame_num", "timestamp"])


def write_stat(frame_num):
    now = datetime.utcnow()
    timestamp = datetime.isoformat(now)

    csv_stat.writerow([frame_num, timestamp])


def urgent_close(*_):
    global new_socket
    global stat_enable

    print("\n=== Завершение имитатора кодировщика FVP-CTVP... ===")

    if stat_enable:
        global stat_file
        stat_file.close()
    new_socket.close()
    sys.exit()


signal.signal(signal.SIGINT, urgent_close)

frame_num = 0


# FPV CTVP VSTR
# --- Заголовок ---
# Версия | байт | 0
# Идентификатор потока | 2 байт | 0
# Тип сегмента | 1 байт | 0 (наземный)
# Тип сообщения | 1 байт | 5 (VSTR)
# Длина сообщения | 8 байт | mlen
# Сообщение | mlen | <...>
# --- Сообщение ---
# Номер кадра | 8 байт | <...>
# Номер сегмента | 2 байт | <...>
# Высота | 4 байт | 720
# Ширина | 4 байт | 1280
# Режим обработки | 1 байт | номер конфигурации
# Режим шифрования | 1 байт | 0
# Длина полезной нагрузки | 8 байт | plen
# Полезная нагрузка | plen | <...>


def pack_packet(message):
    """Запаковать пакет FPV-CTVP."""

    version = 0
    stream_identifier = 0
    segment_type = 0
    message_type = 5
    message_length = len(message)

    packet = struct.pack(">BHBBQ", version,
                         stream_identifier,
                         segment_type,
                         message_type,
                         message_length)
    packet += message

    return packet


def pack_message(frame_num, segment_num, total_segments, cfg_num, payload):
    """Запаковать сообщение FPV-CTVP."""

    height = 720
    width = 1280
    encryption_num = 0
    payload_length = len(payload)

    message = struct.pack(">QHHIIBBQ", frame_num,
                          segment_num,
                          total_segments,
                          height,
                          width,
                          cfg_num,
                          encryption_num,
                          payload_length)
    message += payload

    return message


def pack_packets_from_binary(frame_num, cfg_num, payload):
    """Запаковать множество сообщений из одной полезной нагрузки."""
    global seglen

    packets = []

    seq = 0
    total_segments = math.ceil(len(payload) / seglen)
    pointer = 0

    while pointer < len(payload):
        payload_fragment = payload[pointer:pointer + seglen]
        pointer += seglen

        new_message = pack_message(frame_num, seq, total_segments, cfg_num, payload_fragment)
        new_packet = pack_packet(new_message)
        packets.append(new_packet)

        seq += 1

    return packets


print("=== Запуск кодировщика FPV-CTVP для стенда 1... ===\n")


def show_frame(frame, frame_num):
    this_time = datetime.utcnow()
    text = this_time.strftime("%H:%M:%S:%f")

    height, width, _ = frame.shape
    used_font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3.0
    pos = [width // 2 - 27 * len(text), height // 2]
    color = [(0, 0, 0), (255, 255, 255)]
    thickness = [8, 3]

    for clr, thcn in zip(color, thickness):
        frame = cv2.putText(frame, text, pos, used_font,
                            font_scale, clr, thcn, cv2.LINE_AA)

    text = "Frame: {}".format(frame_num)
    pos = [width // 2 - 27 * len(text), height // 2 + 75]
    for clr, thcn in zip(color, thickness):
        frame = cv2.putText(frame, text, pos, used_font,
                            font_scale, clr, thcn, cv2.LINE_AA)

    cv2.imshow("=== STAND 1 ENCODER ===", frame)
    cv2.waitKey(1)


cap = cv2.VideoCapture(video)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    frame_copy = np.copy(frame)
    show_frame(frame_copy, frame_num)

    write_stat(frame_num)
    start_time = time.time()

    payload = neuro_codec.encode_frame(frame)
    all_packets = pack_packets_from_binary(frame_num, cfg_num, payload)

    print("=== Кадр {} ===".format(frame_num))
    for i, pkt in enumerate(all_packets):
        print("Пакет {}: {}".format(i, len(pkt)))

    for packet in all_packets:
        try:
            new_socket.sendto(packet, socket_address)
        except OSError:
            urgent_close()
            break
        time.sleep(inter_segment_waiter)

    end_time = time.time()

    minus_wait = end_time - start_time
    print("> Время: {} мс".format(round(minus_wait * 1000, 2)))
    real_waiter = waiter - minus_wait
    if real_waiter < 0:
        real_waiter = 0

    time.sleep(real_waiter)

    frame_num += 1

urgent_close()
