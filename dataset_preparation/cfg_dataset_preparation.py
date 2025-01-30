import os
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import cv2
from production_system.production_guardian import ConfigurationGuardian


this_video = "dataset_preparation/25fps.mp4"
# "Clip_3.mov"
dest_dir = "dataset_preparation/cfg14_spbsut_25_full"
# "compression_dataset"  artifacts_dataset
os.makedirs(dest_dir, exist_ok=True)
dest_frames = 10000  # 10 100 1000
basic_size = (1280, 720)
float_mode = True
this_cfg = 14

this_maxsize = 37_580_963_840
cfg_guard = ConfigurationGuardian(this_maxsize, enable_encoder=True, enable_decoder=True)
neuro_codec = cfg_guard.get_configuration(this_cfg)

cap = cv2.VideoCapture(this_video)

length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
print(length)

if length < dest_frames:
    dest_frames = length

if float_mode:
    every_frame = length / dest_frames
    starter = ((length - 1) - (dest_frames - 1) * every_frame) / 2
    count_positions = [int(starter + i * every_frame) for i in range(dest_frames)]
else:
    every_frame = length // dest_frames
    starter = ((length - 1) - (dest_frames - 1) * every_frame) // 2
    count_positions = [starter + i * every_frame for i in range(dest_frames)]

print(count_positions)
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
            print(position1, position2)
            break
    frame = cv2.resize(frame, basic_size, interpolation=cv2.INTER_AREA)

    latent = neuro_codec.encode_frame(frame)
    frame = neuro_codec.decode_frame(latent, dest_height=basic_size[1], dest_width=basic_size[0])

    cv2.imwrite(next_path, frame)

cap.release()
