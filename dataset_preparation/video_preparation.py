import os
import csv
from datetime import datetime
import cv2

in_file = "4x.mp4"
out_file = "record.avi"
frame_size = (1280, 720)

fps = 25
ds = 1 / fps
this_time = 0
video_time = 0

used_font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 3.0
pos = [frame_size[0] // 2, frame_size[1] // 2]
color = [(0, 0, 0), (255, 255, 255)]
thickness = [8, 3]

cap = cv2.VideoCapture(in_file)
fps_source = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
ds_source = 1 / fps_source

out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)


def prepare_frame(frame, frame_num):
    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

    texter = "FRAME {}".format(frame_num)
    dpos = pos.copy()
    dpos[0] -= len(texter) * 27
    dpos[1] -= 25
    for clr, thcn in zip(color, thickness):
        frame = cv2.putText(frame, texter, dpos, used_font, font_scale, clr, thcn, cv2.LINE_AA)

    return frame


frame_num = -1
while True:
    ret, frame = cap.read()

    if not ret:
        break

    video_time += ds_source
    while this_time < video_time:
        frame_num += 1
        frame = prepare_frame(frame, frame_num)
        out.write(frame)
        this_time += ds

        print("> Кадр {}".format(frame_num))

out.release()
cap.release()
