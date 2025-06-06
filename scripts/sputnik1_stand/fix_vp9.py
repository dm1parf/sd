import os
import re
import argparse

frame_dir = "encoder_vp9_hd_25"
extract_format = ".*frame_(\d+).png"
good_format = "frame_{:04d}.png"

parser = argparse.ArgumentParser(prog="Исправитель директорий", description="Исправляет директории файлов")
parser.add_argument('-d', "--frame_dir", dest="frame_dir", type=str, default=frame_dir)
parser.add_argument("--extract_format", dest="extract_format", type=str, default=extract_format)
parser.add_argument("--good_format", dest="good_format", type=str, default=good_format)
args = parser.parse_args()

frame_dir = args.frame_dir
extract_format = args.extract_format
good_format = args.good_format
good_format = os.path.join(frame_dir, good_format)


all_files = [os.path.join(frame_dir, i) for i in os.listdir(frame_dir)]
for file in all_files:
    try:
        number = int(re.match(extract_format, file)[1])
    except TypeError:
        continue
    good_file = good_format.format(number)
    os.rename(file, good_file)




