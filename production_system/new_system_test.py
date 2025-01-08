import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["WORLD_SIZE"] = "1"
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import time
import cv2
import argparse
from production_system.production_guardian import ConfigurationGuardian

# python3.11 production_system/new_system_test.py --cfg 2
# python3.11 scripts/stand1/stand1_encoder.py -v dataset_preparation/25fps.mp4 --cfg 2

arguments = argparse.ArgumentParser(prog="Эмулятор кодера FPV CTVP",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-c", "--cfg", dest="cfg", type=int, default=14,
                       help="Номер конфигурации FPV CTVP")
args = arguments.parse_args()
this_configuration = args.cfg

# ORIN
## 9_961_472_000,  # 9500 Mb
# A100
## 37_580_963_840,  # 35 Gb
this_maxsize = 37_580_963_840  # 35 Gb
guardian_coder = ConfigurationGuardian(this_maxsize, enable_encoder=True, enable_decoder=False)
test_video = "dataset_preparation/25fps.mp4"
neuro_codec = guardian_coder.get_configuration(this_configuration)

cap = cv2.VideoCapture(test_video)
# cv2.namedWindow('ENCODER', cv2.WINDOW_NORMAL)


i = 0
while i < 1000:
    ret, frame = cap.read()
    if not ret:
        break

    a = time.time()
    payload = neuro_codec.encode_frame(frame)
    b = time.time()
    delta = (b - a) * 1000

    print(len(payload), delta)
