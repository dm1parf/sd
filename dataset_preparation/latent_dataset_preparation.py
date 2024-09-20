import os
import torch
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from scripts.fpv_ctvp_emulators.mock_fpv_ctvp_decoder import ConfigurationGuardian

configurations = range(1, 15)

for this_configuration in configurations:
    # this_configuration = 14
    cfg_guard = ConfigurationGuardian()
    neuro_codec = cfg_guard.get_configuration(this_configuration)

    # this_video = "Clip_3.mov"
    this_video = "dataset_preparation/4x.mp4"
    dest_dir = "dataset_preparation/latent_dataset_cfg{}".format(this_configuration)
    # dest_dir = "dataset_preparation/source_dataset"

    os.makedirs(dest_dir, exist_ok=True)
    dest_fps = 25
    basic_size = (1280, 720)
    float_mode = True

    cap = cv2.VideoCapture(this_video)
    length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(length, fps)
    dest_time = 1 / dest_fps
    this_time = 1 / fps
    print(dest_time, this_time)

    i = 0
    timer = i

    k = 0
    k2 = 0


    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            frame = np.copy(frame)

            if not ret:
                break

            if timer >= dest_time:
                timer -= dest_time
                next_name = "{}.latent".format(i)
                # next_name = "{}.jpg".format(i)
                next_path = os.path.join(dest_dir, next_name)

                binary = neuro_codec.encode_frame(frame)

                print(next_name, len(binary))

                with open(next_path, mode='wb') as filer:
                    filer.write(binary)

                # frame = cv2.resize(frame, (1080, 720))
                # cv2.imwrite(next_path, frame)

                i += 1

            timer += this_time

    cap.release()
