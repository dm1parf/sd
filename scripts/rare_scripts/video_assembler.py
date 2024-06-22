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

        current_frame = cv2.imread(this_fn)

        this_datetime = datetime.fromisoformat(record[-1])
        if not old_datetime:
            old_datetime = this_datetime
        # this_delta = (this_datetime - old_datetime).microseconds / 1_000_000
        this_delta = (this_datetime - old_datetime).total_seconds()
        print(this_fn, this_datetime, old_datetime, this_delta)


out.release()
