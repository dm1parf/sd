import argparse
import os
import signal
import socket
import struct
import csv
import time
import sys
import math

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)

cfg_num = 14
hard_port = 6571
ip = "127.0.0.1"
fps = 25
seglen = 1000

arguments = argparse.ArgumentParser(prog="Эмулятор кодера FPV CTVP",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-c", "--cfg", dest="cfg", type=int, default=cfg_num,
                       help="Номер конфигурации FPV CTVP")
arguments.add_argument("-d", "--dest_ip", dest="ip", type=str, default=ip,
                       help="IP-адрес назначения")
arguments.add_argument("-f", "--fps", dest="fps", type=int, default=fps,
                       help="FPS кодера")
arguments.add_argument("-s", "--seglen", dest="seglen", type=int, default=seglen,
                       help="Длина сегментации бинарной последовательности")
args = arguments.parse_args()

cfg_num = args.cfg
ip = args.ip
fps = args.fps
seglen = args.seglen
lat_dir = "dataset_preparation/latent_dataset_cfg{}".format(cfg_num)
waiter = 1 / fps
inter_segment_waiter = 0.001

lat_files = os.listdir(lat_dir)
lens = len(".latent")
lat_nums = [int(i[:-lens]) for i in lat_files]
lat_max = max(lat_nums)

socket_address = (ip, hard_port)
new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# timeout = 5
# new_socket.settimeout(timeout)

stat_enable = False
if stat_enable:
    stat_filename = "stat_mock_fpv_ctvp_coder.csv"
    stat_file = open(stat_filename, 'w', newline='')
    csv_stat = csv.writer(stat_file)
    csv_stat.writerow(["temp"])


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


print("=== Запуск имитатора кодировщика FPV-CTVP... ===\n")

while frame_num <= lat_max:
    lat_file = "{}.latent".format(frame_num)
    lat_file = os.path.join(lat_dir, lat_file)

    start_time = time.time()

    with open(lat_file, mode="rb") as lf:
        payload = lf.read()

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
    real_waiter = waiter - minus_wait
    if real_waiter < 0:
        real_waiter = 0

    time.sleep(real_waiter)

    frame_num += 1

urgent_close()
