import time
import cv2
import argparse


video = "0"
arguments = argparse.ArgumentParser(prog="Эмулятор кодера FPV CTVP",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-v", "--video", dest="video", type=str, default=video,
                       help="Ссылка на RTSP-стрим")
args = arguments.parse_args()

video = args.video


sleep_time = 0.1
cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(sleep_time)
        continue

    cv2.imshow("===", frame)
    cv2.waitKey(1)
