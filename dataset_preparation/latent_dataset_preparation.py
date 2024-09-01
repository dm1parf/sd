import os
import torch
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from utils.workers import (WorkerASDummy, WorkerQuantLinear, WorkerAutoencoderKL_F16, WorkerCompressorDummy,
                           WorkerAutoencoderKL_F4, WorkerCompressorJpegXL, WorkerCompressorJpegXR,
                           WorkerCompressorAvif, WorkerQuantPower)

as_ = WorkerASDummy()
# quant = WorkerQuantLinear()
# quant = WorkerQuantLinear(pre_quant="scale", nsd=1)
quant = WorkerQuantPower(pre_quant="scale", nsd=1)
vae = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml", ckpt_path="dependence/ckpt/kl-f16.ckpt")
# vae = WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml", ckpt_path="dependence/ckpt/kl-f4.ckpt")
quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")  # AutoencoderKL_F4
# compress = WorkerCompressorDummy()
compress = WorkerCompressorJpegXR(70)

# this_video = "Clip_3.mov"
this_video = "dataset_preparation/4x.mp4"
dest_dir = "dataset_preparation/latent_dataset_cfg12"
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
            next_path = os.path.join(dest_dir, next_name)

            frame = frame.reshape(1, *frame.shape)

            image, _ = as_.prepare_work(frame)
            latent, _ = vae.encode_work(image)
            (quant_latent, _), _ = quant.quant_work(latent)
            binary, _ = compress.compress_work(quant_latent)

            print(next_name, len(binary))

            with open(next_path, mode='wb') as filer:
                filer.write(binary)

            i += 1

        timer += this_time

cap.release()
