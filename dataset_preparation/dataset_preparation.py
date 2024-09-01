import os
import cv2

# this_video = "Clip_3.mov"
this_video = "7.mp4"
dest_dir = "artifact10"
os.makedirs(dest_dir, exist_ok=True)
dest_frames = 1000  # 10 1000
basic_size = (1280, 720)
float_mode = True

cap = cv2.VideoCapture(this_video)
length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))

if length < dest_frames:
    dest_frames = length

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

    if not ret:
        position1 = position
        position2 = position
        while True:
            position1 -= 1
            position2 += 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, position1)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position2)
                ret, frame = cap.read()

                if not ret:
                    continue
            break

    frame = cv2.resize(frame, basic_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(next_path, frame)

cap.release()
