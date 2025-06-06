import cv2
import os
import argparse


frame_dir = "encoder_h266_hd_25"
bad_format = ".bmp"
good_format = ".png"


parser = argparse.ArgumentParser(prog="Исправитель директорий", description="Исправляет директории файлов")
parser.add_argument('-d', "--frame_dir", dest="frame_dir", type=str, default=frame_dir)
parser.add_argument('-f', "--bad_format", dest="bad_format", type=str, default=bad_format)
args = parser.parse_args()

frame_dir = args.frame_dir
bad_format = args.bad_format


list_files = [os.path.join(frame_dir, i) for i in os.listdir(frame_dir) if i.endswith(bad_format)]
for bad_file in list_files:
    good_file = os.path.splitext(bad_file)[0] + good_format

    frame = cv2.imread(bad_file)
    os.remove(bad_file)
    cv2.imwrite(good_file, frame)

    print(bad_file)
    print(good_file)
    print('----')

