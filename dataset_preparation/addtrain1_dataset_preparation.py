import os
import sys
sys.path.append(os.getcwd())
import torch
import cv2
import numpy as np
from utils.workers import WorkerASMoveDistribution, WorkerQuantLinear, WorkerAutoencoderKL_F4, WorkerCompressorJpegXL
from scripts.fpv_ctvp_emulators.mock_fpv_ctvp_decoder import NeuroCodec

dataset = "materials/dataset"
# "materials/dataset" "dataset_preparation/spbsut_dataset_1000"
output_dataset = "materials/dataset_addtrain1"
# "materials/dataset_addtrain1" "dataset_preparation/spbsut_dataset_addtrain1"
os.makedirs(output_dataset, exist_ok=True)

true_extensions = [".jpg"]

as_ = WorkerASMoveDistribution()
kl_f4 = WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml",
                                 ckpt_path="dependence/ckpt/kl-f4.ckpt")
nominal_type = torch.float32
kl_f4._nominal_type = nominal_type
quant = WorkerQuantLinear(pre_quant="bitround", nsd=1)
quant.adjust_params(autoencoder_worker="AutoencoderKL_F4")
compress = WorkerCompressorJpegXL(quality=65)
codec = NeuroCodec(as_=as_, vae=kl_f4, quant=quant, compressor=compress)

with torch.no_grad():
    for root, dirs, files in os.walk(dataset):
        for file in files:
            from_filepath = os.path.join(root, file)
            pre_path = os.path.relpath(from_filepath, dataset)
            to_filepath = os.path.join(output_dataset, pre_path)
            to_filepath = os.path.splitext(to_filepath)[0] + ".lpp"
            to_dir = os.path.split(to_filepath)[0]
            os.makedirs(to_dir, exist_ok=True)

            image = cv2.imread(from_filepath)
            latent = codec.encode_frame(image)

            decompress_latent, _ = compress.decompress_work(latent, dest_shape=kl_f4.z_shape, dest_type=torch.uint8)
            dequant_latent, _ = quant.dequant_work(decompress_latent, dest_type=nominal_type)

            numpy_lat = dequant_latent.cpu().numpy()
            byter = numpy_lat.tobytes(order='C')

            with open(to_filepath, mode='wb') as wf:
                wf.write(byter)
