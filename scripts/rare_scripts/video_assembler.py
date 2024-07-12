import os
import csv
from datetime import datetime
import cv2

records_dir = "decoder_records"
records_csv = "stat_decoder.csv"
file_mask = "{}.png"
out_file = "record.avi"
frame_size = (1280, 720)

fps = 30
ds = 1 / fps
this_time = 0
old_datetime = None
current_frame = None

used_font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
pos = [(frame_size[0]-150, 20+15*i) for i in range(1, 4)]
color = [(0, 0, 0), (255, 255, 255)]
thickness = [2, 1]


print(ds)
out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
with open(records_csv, mode='r', encoding="utf-8") as rf:
    csvr = csv.reader(rf, delimiter=",")
    next(csvr)
    for record in csvr:

        if current_frame is not None:
            while this_time < this_delta:
                out.write(current_frame)
                this_time += ds

        this_fn = os.path.join(records_dir, file_mask.format(record[0]))
        bin_size = "{} bytes".format(record[1])
        fps = "{} FPS".format(record[2])
        fps = fps.replace(".", ",")
        kbps = "{} kbps".format(record[3])
        layer_texts = [bin_size, fps, kbps]

        current_frame = cv2.imread(this_fn)
        for ps, text in zip(pos, layer_texts):
            for clr, thcn in zip(color, thickness):
                current_frame = cv2.putText(current_frame, text, ps, used_font,
                                            font_scale, clr, thcn, cv2.LINE_AA)

        this_datetime = datetime.fromisoformat(record[-1])
        if not old_datetime:
            old_datetime = this_datetime
        # this_delta = (this_datetime - old_datetime).microseconds / 1_000_000
        this_delta = (this_datetime - old_datetime).total_seconds()
        print(this_fn, this_datetime, old_datetime, this_delta)


out.release()
