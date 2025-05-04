import os
import sys

sys.path.append(os.getcwd())
import torch
import cv2
import numpy as np
from utils.workers import WorkerASMoveDistribution, WorkerQuantLinear, WorkerAutoencoderKL_F4, WorkerCompressorJpegXL
from scripts.fpv_ctvp_emulators.mock_fpv_ctvp_decoder import NeuroCodec
from dependence.util import instantiate_from_config
from omegaconf import OmegaConf

# dataset = "materials/dataset" "dataset_preparation/spbsut_dataset_1000"
dataset = "dataset_preparation/spbsut_dataset_1000"
output_dataset = "dataset_preparation/spbsut_dataset_addtrain1"
# "materials/dataset_addtrain1" "dataset_preparation/spbsut_dataset_addtrain1"
os.makedirs(output_dataset, exist_ok=True)

true_extensions = [".jpg"]

as_ = WorkerASMoveDistribution()
kl_f4 = WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml",
                              ckpt_path="dependence/ckpt/kl-f4.ckpt")
kl_f4._nominal_type = torch.float32
quant = WorkerQuantLinear(pre_quant="bitround", nsd=1)
quant.adjust_params(autoencoder_worker="AutoencoderKL_F4")
compress = WorkerCompressorJpegXL(quality=65)
# codec = NeuroCodec(as_=as_, vae=kl_f4, quant=quant, compressor=compress)

config = OmegaConf.load("dependence/config/kl-f4.yaml")
model = instantiate_from_config(config.model)
pl_sd = torch.load("dependence/ckpt/kl-f4.ckpt", map_location="cpu")
sd = pl_sd["state_dict"]
model.load_state_dict(sd, strict=False)
model.forward = model.decode
model = model.type(torch.float32).cuda()
model.eval()

with torch.no_grad():
    for root, dirs, files in os.walk(dataset):
        for file in files:
            from_filepath = os.path.join(root, file)
            pre_path = os.path.relpath(from_filepath, dataset)
            to_filepath = os.path.join(output_dataset, pre_path)
            to_filepath = os.path.splitext(to_filepath)[0] + ".lpp"

            with open(to_filepath, mode='rb') as rf:
                compressed_bytes = rf.read()
            lpp_tensor = torch.frombuffer(compressed_bytes, dtype=torch.float32)
            lpp_tensor = lpp_tensor.reshape(kl_f4.z_shape)
            lpp_tensor = lpp_tensor.to("cuda")

            # img, _ = kl_f4.decode_work(lpp_tensor)
            img = model(lpp_tensor)
            image, _ = as_.restore_work(img)
            cv2.imshow("===", image)
            cv2.waitKey(50)
