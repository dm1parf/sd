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
data_loader = config.get_data_loader()
batch_size = data_loader.batch_size
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


def split_batch_numpy(arr: np.ndarray, this_batch_size: int) -> list[np.ndarray]:
    all_arrs = [i[0] for i in np.split(arr, this_batch_size, axis=0)]
    return all_arrs


def split_batch_torch(arr: torch.Tensor, this_batch_size: int) -> list[torch.Tensor]:
    all_arrs = list(torch.chunk(arr, this_batch_size, dim=0))
    return all_arrs


def combine_batch_numpy(all_arrs: list[np.ndarray]) -> np.ndarray:

    true_size = [1] + list(all_arrs[0].shape)
    all_arrs = [np.reshape(i, true_size) for i in all_arrs]
    arr = np.concatenate(all_arrs, axis=0)
    return arr


def combine_batch_torch(all_arrs: list[torch.Tensor]) -> torch.Tensor:

    arr = torch.cat(all_arrs, dim=0)
    return arr


signal.signal(signal.SIGINT, sudden_shutdown)

with torch.no_grad():
    # for id_, (name, image) in enumerate(dataset):
    for id_, (name, image) in enumerate(data_loader):
        this_batch_size = image.shape[0]

        image = image.cpu()
        start_numpy = image.numpy()
        start_numpy = np.moveaxis(start_numpy, 1, 3)

        all_images = split_batch_numpy(start_numpy, this_batch_size)
        start_numpy = []
        start_shape = []
        beginning_time = time.time()
        as_prepare_time = []
        for i, one_image in enumerate(all_images):
            one_image = cv2.cvtColor(one_image, cv2.COLOR_RGB2BGR)
            start_numpy.append(one_image)
            start_shape.append(list(one_image.shape))
            one_image = np.copy(one_image)
            all_images[i], as_prep_time = as_.prepare_work(one_image)
            as_prepare_time.append(as_prep_time)
        image = combine_batch_torch(all_images)

        encoding_time = []
        if vae:
            image, enc_time = vae.encode_work(image)
            dest_shape = vae.z_shape
        else:
            enc_time = 0
            dest_shape = list(image.shape)
            dest_shape[0] = 1
        encoding_time = [enc_time / this_batch_size] * this_batch_size
        latent_size = reduce(lambda x, y: x * y, list(image.shape))
        latent_size = [latent_size // this_batch_size] * this_batch_size

        quant_time = []
        if quant:
            (image, quant_params), q_time = quant.quant_work(image)
            dest_type = torch.uint8
        else:
            if vae:
                dest_type = vae.nominal_type
            else:
                dest_type = torch.float16
            q_time = 0
        quant_time = [q_time / this_batch_size] * this_batch_size

        all_images = split_batch_torch(image, this_batch_size)
        compress_time = []
        decompress_time = []
        min_size = []
        torch.cuda.synchronize()
        transmit_timepoint = time.time()
        total_coder_time = transmit_timepoint - beginning_time
        for i, one_image in enumerate(all_images):
            new_frac_beginning = time.time()
            one_image, comp_time = compressor.compress_work(one_image)
            compress_time.append(comp_time)
            min_size.append(len(one_image))
            new_frac = time.time() - new_frac_beginning
            transmit_timepoint += new_frac

            one_image, decomp_time = compressor.decompress_work(one_image, dest_shape, dest_type)
            decompress_time.append(decomp_time)
            all_images[i] = one_image
        total_coder_time = [total_coder_time / this_batch_size] * this_batch_size

        image = combine_batch_torch(all_images)

        dequant_time = []
        if quant:
            image, deq_time = quant.dequant_work(image)
        else:
            deq_time = 0
        dequant_time = [deq_time / this_batch_size] * this_batch_size
        decoding_time = []
        if vae:
            image, dec_time = vae.decode_work(image)
        else:
            dec_time = 0
        decoding_time = [dec_time / this_batch_size] * this_batch_size

        predictor_time = [0] * this_batch_size

        all_images = split_batch_torch(image, this_batch_size)
        end_numpy = []
        as_restore_time = []
        is_black = []
        superresolution_time = []
        for i, one_image in enumerate(all_images):
            one_image, as_rest_time = as_.restore_work(one_image)
            as_restore_time.append(as_rest_time)

            if one_image.any():
                is_black.append(False)
            else:
                is_black.append(True)

            one_image, superres_time = sr.sr_work(one_image, dest_size=start_shape[i][::-1][1:])
            end_numpy.append(one_image)
            superresolution_time.append(superres_time)
            all_images[i] = np.copy(one_image)

        torch.cuda.synchronize()
        end_time = time.time()
        total_decoder_time = [(end_time - transmit_timepoint) / this_batch_size] * this_batch_size
        total_time = [(end_time - beginning_time) / this_batch_size] * this_batch_size

        psnr = []
        ssim = []
        mse = []
        sum_time = []
        bitrate = []
        for i, one_image in enumerate(all_images):
            if is_black[i]:
                psnr.append(np.nan)
                ssim.append(np.nan)
                mse.append(np.nan)
            else:
                psnr.append(stat_mng.psnr_metric(start_numpy[i], end_numpy[i]))
                ssim.append(stat_mng.ssim_metric(start_numpy[i], end_numpy[i]))
                mse.append(stat_mng.mse_metric(start_numpy[i], end_numpy[i]))

            if imwrite:
                new_name = os.path.splitext(name[i])[0] + ".jpg"
                new_name = os.path.join(imwrite_path, new_name)
                end_numpy = cv2.cvtColor(end_numpy, cv2.COLOR_BGR2RGB)
                cv2.imwrite(new_name, end_numpy)

            sum_time_ = encoding_time[i] + quant_time[i] + compress_time[i]
            if sum_time == 0:
                bitrate.append(np.inf)
            else:
                bitrate.append(512 * 512 * 3 * 8 / sum_time_)
            sum_time.append(sum_time_)

        for i, one_image in enumerate(all_images):
            real_id = id_ * batch_size + i

            stat_mng.write_stat([real_id, name[i],
                                 psnr[i], mse[i], ssim[i],
                                 latent_size[i], min_size[i],
                                 as_prepare_time[i], as_restore_time[i],
                                 encoding_time[i], decoding_time[i],
                                 quant_time[i], dequant_time[i],
                                 compress_time[i], decompress_time[i],
                                 superresolution_time[i], predictor_time[i],
                                 total_coder_time[i], total_decoder_time[i], total_time[i],
                                 bitrate[i]])

            if (real_id % progress_check) == 0:
                print("=== Прогресс: {}% ({}/{})".format(round(real_id / max_entries * 100, 3), real_id, max_entries))
            if real_id >= max_entries:
                break

stat_mng.cleanup()
