import os
import time
import signal
from functools import reduce
import cv2
import torch
import torchvision.transforms.functional
import numpy as np
from outer_models.test_scripts.utils.config import ConfigManager

config_path = "outer_models/test_scripts/experiment_config.ini"
config = ConfigManager(config_path)

stat_mng = config.get_stat_mng()
dataset = config.get_dataset()
dataset_len = len(dataset)

max_entries = config.get_max_entries()
if max_entries == 0:
    max_entries = dataset_len
else:
    max_entries = min(max_entries, dataset_len)
progress_check = config.get_progress_check()

imwrite, imwrite_path = config.get_imwrite_params()
if imwrite:
    os.makedirs(imwrite_path, exist_ok=True)

vae = config.get_autoencoder_worker()
quant = config.get_quant_worker()
compressor = config.get_compress_worker()
sr = config.get_sr_worker()


def sudden_shutdown(*_, **__):
    global stat_mng
    stat_mng.cleanup()
    exit()


signal.signal(signal.SIGINT, sudden_shutdown)

for id_, (name, image) in enumerate(dataset):
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
    latent_size = reduce(lambda x, y: x * y, list(image.shape))
    (image, quant_params), quant_time = quant.quant_work(image)
    if compressor:
        image, compress_time = compressor.compress_work(image)
    min_size = len(image)

    torch.cuda.synchronize()
    transmit_timepoint = time.time()
    total_coder_time = transmit_timepoint - beginning_time

    if compressor:
        image, decompress_time = compressor.decompress_work(image)
    image, dequant_time = quant.dequant_work(image)
    image, decoding_time = vae.decode_work(image)
    image *= 255.0
    image = image.to(torch.uint8)
    image = image.reshape(3, 512, 512)

    predictor_time = 0

    image = image.cpu()
    end_numpy = image.numpy()
    end_numpy = np.moveaxis(end_numpy, 0, 2)
    end_numpy, superresolution_time = sr.sr_work(end_numpy, dest_size=start_shape[3:0:-1])

    torch.cuda.synchronize()
    end_time = time.time()
    total_decoder_time = end_time - transmit_timepoint
    total_time = end_time - beginning_time

    psnr = stat_mng.psnr_metric(start_numpy, end_numpy)
    ssim = stat_mng.ssim_metric(start_numpy, end_numpy)
    mse = stat_mng.mse_metric(start_numpy, end_numpy)

    if imwrite:
        new_name = os.path.splitext(name)[0] + ".png"
        new_name = os.path.join(imwrite_path, new_name)
        end_numpy = cv2.cvtColor(end_numpy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_name, end_numpy)

    min_size = 0
    quant_time = 0
    dequant_time = 0
    compress_time = 0
    decompress_time = 0
    stat_mng.write_stat([id_, name,
                         psnr, ssim, mse,
                         latent_size, min_size,
                         encoding_time, decoding_time,
                         quant_time, dequant_time,
                         compress_time, decompress_time,
                         superresolution_time, predictor_time,
                         total_coder_time, total_decoder_time, total_time])

    if (id_ % progress_check) == 0:
        print("=== Прогресс: {}% ({}/{})".format(round(id_ / max_entries * 100, 3), id_, max_entries))
    if id_ >= max_entries:
        break

stat_mng.cleanup()
