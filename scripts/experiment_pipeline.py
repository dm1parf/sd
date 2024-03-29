import os
import sys
import time
import signal
from functools import reduce
import cv2
import torch
import torchvision.transforms.functional
import numpy as np
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager


config_path = os.path.join("scripts", "experiment_config.ini")
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
as_ = config.get_as_worker()


def sudden_shutdown(*_, **__):
    global stat_mng
    stat_mng.cleanup()
    exit()


signal.signal(signal.SIGINT, sudden_shutdown)

with torch.no_grad():
    for id_, (name, image) in enumerate(dataset):
        image = image.cpu()
        start_numpy = image.numpy()

        start_numpy = np.moveaxis(start_numpy, 0, 2)
        start_numpy = cv2.cvtColor(start_numpy, cv2.COLOR_RGB2BGR)
        start_shape = list(start_numpy.shape)
        as_numpy = np.copy(start_numpy)
        beginning_time = time.time()
        image, as_prepare_time = as_.prepare_work(as_numpy)

        if vae:
            image, encoding_time = vae.encode_work(image)
            dest_shape = vae.z_shape
        else:
            encoding_time = 0
            dest_shape = list(image.shape)
        latent_size = reduce(lambda x, y: x * y, list(image.shape))
        if quant:
            (image, quant_params), quant_time = quant.quant_work(image)
            dest_type = torch.uint8
        else:
            if vae:
                dest_type = vae.nominal_type
            else:
                dest_type = torch.float16
            quant_time = 0

        image, compress_time = compressor.compress_work(image)
        min_size = len(image)

        torch.cuda.synchronize()
        transmit_timepoint = time.time()
        total_coder_time = transmit_timepoint - beginning_time

        image, decompress_time = compressor.decompress_work(image, dest_shape, dest_type)
        if quant:
            image, dequant_time = quant.dequant_work(image)
        else:
            dequant_time = 0
        if vae:
            image, decoding_time = vae.decode_work(image)
        else:
            decoding_time = 0

        predictor_time = 0
        end_numpy, as_restore_time = as_.restore_work(image)

        if end_numpy.any():
            is_black = False
        else:
            is_black = True
        end_numpy, superresolution_time = sr.sr_work(end_numpy, dest_size=start_shape[::-1][1:])

        torch.cuda.synchronize()
        end_time = time.time()
        total_decoder_time = end_time - transmit_timepoint
        total_time = end_time - beginning_time

        if is_black:
            psnr = np.nan
            ssim = np.nan
            mse = np.nan
        else:
            psnr = stat_mng.psnr_metric(start_numpy, end_numpy)
            ssim = stat_mng.ssim_metric(start_numpy, end_numpy)
            mse = stat_mng.mse_metric(start_numpy, end_numpy)

        if imwrite:
            new_name = os.path.splitext(name)[0] + ".jpg"
            new_name = os.path.join(imwrite_path, new_name)
            end_numpy = cv2.cvtColor(end_numpy, cv2.COLOR_BGR2RGB)
            cv2.imwrite(new_name, end_numpy)

        bitrate = 512 * 512 * 3 * 8 / (encoding_time + quant_time + compress_time)
        stat_mng.write_stat([id_, name,
                             psnr, mse, ssim,
                             latent_size, min_size,
                             as_prepare_time, as_restore_time,
                             encoding_time, decoding_time,
                             quant_time, dequant_time,
                             compress_time, decompress_time,
                             superresolution_time, predictor_time,
                             total_coder_time, total_decoder_time, total_time,
                             bitrate])

        if (id_ % progress_check) == 0:
            print("=== Прогресс: {}% ({}/{})".format(round(id_ / max_entries * 100, 3), id_, max_entries))
        if id_ >= max_entries:
            break

stat_mng.cleanup()
