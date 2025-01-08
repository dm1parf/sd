import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import argparse
import signal
import socket
import csv
import time
import math
import shutil
import cv2
from datetime import datetime
from production_system.production_guardian import ConfigurationGuardian
from production_system.packet_manager import PacketManager


cfg_num = 14
hard_port = 6571
ip = "172.16.204.14"
fps = 100500
seglen = 1000
video = "0"

arguments = argparse.ArgumentParser(prog="Эмулятор кодера FPV CTVP",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-c", "--cfg", dest="cfg", type=int, default=cfg_num,
                       help="Номер конфигурации FPV CTVP")
arguments.add_argument("-v", "--video", dest="video", type=str, default=video,
                       help="Ссылка на RTSP-стрим")
arguments.add_argument("-d", "--dest_ip", dest="ip", type=str, default=ip,
                       help="IP-адрес назначения")
arguments.add_argument("-f", "--fps", dest="fps", type=int, default=fps,
                       help="FPS кодера")
arguments.add_argument("-s", "--seglen", dest="seglen", type=int, default=seglen,
                       help="Длина сегментации бинарной последовательности")
arguments.add_argument("--device", dest="device", type=int, default=-1,
                       help="Устройство")
arguments.add_argument('--record', dest="record", action='store_true', default=False)
arguments.add_argument('--fullhd', dest="fullhd", action='store_true', default=False)
args = arguments.parse_args()

if args.device != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
cfg_num = args.cfg
ip = args.ip
fps = args.fps
seglen = args.seglen
video = args.video
# lat_dir = "dataset_preparation/latent_dataset_cfg{}".format(cfg_num)
lat_dir = "source_frames"
waiter = 1 / fps
inter_segment_waiter = 0.001
record = args.record
if record:
    if os.path.isdir(lat_dir):
        shutil.rmtree(lat_dir, ignore_errors=True)
    os.makedirs(lat_dir)
fullHD_mode = args.fullhd
this_maxsize = 37_580_963_840
if fullHD_mode:
    height = 1080
    width = 1920
else:
    height = 720
    width = 1280

cfg_guard = ConfigurationGuardian(this_maxsize, enable_encoder=True, enable_decoder=False)
neuro_codec = cfg_guard.get_configuration(cfg_num)
pck_mgr = PacketManager(0)

socket_address = (ip, hard_port)
new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    cv2.destroyAllWindows()
    if stat_enable:
        global stat_file
        stat_file.close()
    new_socket.close()
    sys.exit()


signal.signal(signal.SIGINT, urgent_close)

frame_num = 0


def pack_packets_from_binary(frame_num, cfg_num, payload):
    """Запаковать множество сообщений из одной полезной нагрузки."""
    global seglen

    packets = []

    seq = 0
    total_segments = math.ceil(len(payload) / seglen)
    pointer = 0
    message_type = 5

    while pointer < len(payload):
        payload_fragment = payload[pointer:pointer + seglen]
        pointer += seglen

        new_message = pck_mgr.pack_vstr(frame_num, height, width, seq, total_segments, cfg_num, payload_fragment)
        new_packet = pck_mgr.pack_packet(new_message, message_type)
        packets.append(new_packet)

        seq += 1

    return packets


print("=== Запуск кодировщика FPV-CTVP для стенда 1... ===\n")


cap = cv2.VideoCapture(video)
cv2.namedWindow('ENCODER', cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()
    ret, frame = cap.read()

    cv2.imshow("ENCODER", frame)
    cv2.waitKey(1)
    if record:
        write_path = os.path.join(lat_dir, str(frame_num) + ".png")
        cv2.imwrite(write_path, frame)
    write_stat(frame_num)

    payload = neuro_codec.encode_frame(frame)
    all_packets = pack_packets_from_binary(frame_num, cfg_num, payload)

    print("=== Кадр {} ===".format(frame_num))
    for i, pkt in enumerate(all_packets):
        print("Пакет {}: {}".format(i, len(pkt)))

    for packet in all_packets:
        try:
            new_socket.sendto(packet, socket_address)
        except OSError as err:
            print(err)
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
