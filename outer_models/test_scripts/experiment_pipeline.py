import time
from functools import reduce
import cv2
import torch
import torchvision.transforms.functional
import numpy as np
import cv2
from outer_models.test_scripts.utils.statistics import StatisticsManager
from outer_models.test_scripts.utils.uav_dataset import UAVDataset
from outer_models.test_scripts.utils.workers import (WorkerAutoencoderVQ_F16, WorkerQuantLogistics,
                                                     WorkerCompressorDeflated)

stat_mng = StatisticsManager("statistics.csv")

dataset_path = "outer_models/test_scripts/dataset"
dataset = UAVDataset(dataset_path, name_output=True)
dataset_len = len(dataset)

vae = WorkerAutoencoderVQ_F16(config_path="outer_models/config/vq-f16.yaml", ckpt_path="outer_models/ckpt/vq-f16.ckpt")
quant = WorkerQuantLogistics()
compressor = WorkerCompressorDeflated()

for id, (name, image) in enumerate(dataset):
    image = image.cpu()
    start_numpy = image.numpy()
    start_numpy = np.moveaxis(start_numpy, 0, 2)
    beginning_time = time.time()

    image = image.cuda()
    image = image.to(torch.float16)
    image /= 255.0
    start_shape = list(image.shape)
    image = image.reshape(1, *start_shape)

    image = torchvision.transforms.functional.resize(image, [512, 512])
    image, encoding_time = vae.encode_work(image)
    latent_size = reduce(lambda x, y: x*y, list(image.shape))
    (image, quant_params), quant_time = quant.quant_work(image)
    image, compress_time = compressor.compress_work(image)
    min_size = len(image)

    torch.cuda.synchronize()
    transmit_timepoint = time.time()
    total_coder_time = transmit_timepoint - beginning_time

    image, decompress_time = compressor.decompress_work(image)
    image, dequant_time = quant.dequant_work(image)
    image, decoding_time = vae.decode_work(image)
    image *= 255.0
    image = image.to(torch.uint8)
    image = image.reshape(3, 512, 512)
    image = torchvision.transforms.functional.resize(image, start_shape[1:])

    superresolution_time = 0
    predictor_time = 0

    torch.cuda.synchronize()
    end_time = time.time()
    total_decoder_time = end_time - transmit_timepoint
    total_time = end_time - beginning_time

    image = image.cpu()
    end_numpy = image.numpy()
    end_numpy = np.moveaxis(end_numpy, 0, 2)
    cv2.imwrite(f"{id}.png", end_numpy)
    psnr = stat_mng.psnr_metric(start_numpy, end_numpy)
    ssim = stat_mng.ssim_metric(start_numpy, end_numpy)
    mse = stat_mng.mse_metric(start_numpy, end_numpy)

    stat_mng.write_stat([id, name,
                        psnr, ssim, mse,
                        latent_size, min_size,
                        encoding_time, decoding_time,
                        quant_time, dequant_time,
                        compress_time, decompress_time,
                        superresolution_time, predictor_time,
                        total_coder_time, total_decoder_time, total_time])

    if id > 5:
        break

stat_mng.cleanup()
