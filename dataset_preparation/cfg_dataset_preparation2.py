import os
import sys
sys.path.append(os.getcwd())
import torch
import cv2
import numpy as np
from utils.workers import WorkerASDummy, WorkerQuantLinear, WorkerAutoencoderKL_F16, WorkerCompressorDummy
from scripts.stand2.stand2_decoder import ConfigurationGuardian


dataset = "materials/dataset"
output_dataset = "materials/dataset_cfg1"
os.makedirs(output_dataset, exist_ok=True)
this_cfg = 1
basic_size = (512, 512)

true_extensions = [".jpg", ".png", ".bmp"]

cfg_guard = ConfigurationGuardian()
neuro_codec = cfg_guard.get_configuration(this_cfg)


for root, dirs, files in os.walk(dataset):
    for file in files:
        from_filepath = os.path.join(root, file)
        pre_path = os.path.relpath(from_filepath, dataset)
        to_filepath = os.path.join(output_dataset, pre_path)
        to_filepath = os.path.splitext(to_filepath)[0] + ".jpg"
        to_dir = os.path.split(to_filepath)[0]
        os.makedirs(to_dir, exist_ok=True)

        frame = cv2.imread(from_filepath)
        latent = neuro_codec.encode_frame(frame)
        frame = neuro_codec.decode_frame(latent, dest_height=basic_size[1], dest_width=basic_size[0])

        cv2.imwrite(to_filepath, frame)
