import os
import cv2

this_video = "Clip_3.mov"
dest_dir = "compression_dataset"
os.makedirs(dest_dir, exist_ok=True)
dest_frames = 100  # 10 1000
basic_size = (1280, 720)
float_mode = True

cap = cv2.VideoCapture(this_video)
length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
if float_mode:
    every_frame = length / dest_frames
    starter = ((length-1) - (dest_frames-1)*every_frame)/2
    count_positions = [int(starter + i*every_frame) for i in range(dest_frames)]
else:
    every_frame = length // dest_frames
    starter = ((length-1) - (dest_frames-1)*every_frame)//2
    count_positions = [starter + i*every_frame for i in range(dest_frames)]

for i, position in enumerate(count_positions):
    next_path = os.path.join(dest_dir, str(i) + ".jpg")
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    ret, frame = cap.read()
    frame = cv2.resize(frame, basic_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(next_path, frame)

cap.release()
