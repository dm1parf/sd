import re
import cv2
import easyocr
import numpy as np
import argparse


source_video = "source_video.mp4"
reger = "(\d{2}):(\d{2}):(\d{3})"
reader = easyocr.Reader(lang_list=["en"], gpu=True)

arguments = argparse.ArgumentParser(prog="OCR-считыватель задержек",
                                    description="Сделано для испытаний задержек канала.")
arguments.add_argument("-v", "--video", dest="video", type=str, default=source_video,
                       help="Файл видео")
args = arguments.parse_args()
source_video = args.video


def get_value(strer):
    matcher = re.search(reger, strer)
    if matcher is None:
        return None

    try:
        minutes = int(matcher[1])
        seconds = int(matcher[2])
        milliseconds = int(matcher[3])
    except:
        return None

    seconds += minutes * 60
    milliseconds += seconds * 1000

    return milliseconds


def string_fix(strer):
    strer = strer.replace('.', ':')
    strer = strer.replace(',', ':')
    strer = strer.replace(';', ':')
    strer = strer.replace(' ', '')

    return strer


def get_frame_data(frame):
    results = reader.readtext(frame)

    frame_data = []
    for i in results:
        this_texter = i[-2]
        this_texter = string_fix(this_texter)
        ms_value = get_value(this_texter)

        if not ms_value:
            # DEBUG
            print("!!!", this_texter)
            continue

        frame_data.append(ms_value)

    if len(frame_data) != 2:
        return None

    frame_data.sort()

    return frame_data


all_frame_data = []

cap = cv2.VideoCapture(source_video)
length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))

i = 0
while True:
    i += 1

    ret, frame = cap.read()
    if not ret:
        break

    result = get_frame_data(frame)

    if result not in all_frame_data:
        all_frame_data.append(result)

    print("{} / {}".format(i, length))

all_latencies = [abs(x[0] - x[1]) for x in all_frame_data]

latencies = np.array(all_latencies, dtype=np.float32)
K1 = np.quantile(latencies, 0.25)
K3 = np.quantile(latencies, 0.75)
DK = K3 - K1
min_latency = K1 - 1.5 * DK
max_latency = K3 + 1.5 * DK
latencies = latencies[latencies > min_latency]
latencies = latencies[latencies < max_latency]

mean_latency = latencies.mean()

print("=== !!! ===")
print("Mean Latency: {} ms".format(mean_latency))
print("=== !!! ===")

