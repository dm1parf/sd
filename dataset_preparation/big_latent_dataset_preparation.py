import os
import sys
sys.path.append(os.getcwd())
import torch
import cv2
import numpy as np
from utils.workers import WorkerASDummy, WorkerQuantLinear, WorkerAutoencoderKL_F16, WorkerCompressorDummy
from scripts.fpv_ctvp_emulators.mock_fpv_ctvp_decoder import NeuroCodec

dataset = "materials/dataset"
output_dataset = "materials/dataset_kl-f14_latent"
os.makedirs(output_dataset, exist_ok=True)

true_extensions = [".jpg", ".png", ".bmp"]

as_ = WorkerASDummy()
kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                 ckpt_path="dependence/ckpt/kl-f16.ckpt")
quant = WorkerQuantLinear()
quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
compress = WorkerCompressorDummy()
codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

for root, dirs, files in os.walk(dataset):
    for file in files:
        from_filepath = os.path.join(root, file)
        pre_path = os.path.relpath(from_filepath, dataset)
        to_filepath = os.path.join(output_dataset, pre_path)
        to_filepath = os.path.splitext(to_filepath)[0] + ".latent"
        to_dir = os.path.split(to_filepath)[0]
        os.makedirs(to_dir, exist_ok=True)

        image = cv2.imread(from_filepath)
        latent = codec.encode_frame(image)

        with open(to_filepath, mode='wb') as wf:
            wf.write(latent)
