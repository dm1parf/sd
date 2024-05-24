import os
import sys
import time
import collections
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


temp_path = "temp/"

os.makedirs(temp_path, exist_ok=True)
config_path = os.path.join("scripts", "rare_scripts", "predictor_test_pipeline1_config.ini")
config = ConfigManager(config_path)

dataset = config.get_dataset()
basic_size = config.get_basic_size()
dataset_len = len(dataset)

max_entries = config.get_max_entries()
if max_entries == 0:
    max_entries = dataset_len
else:
    max_entries = min(max_entries, dataset_len)
progress_check = config.get_progress_check()

vae = config.get_autoencoder_worker()
quant = config.get_quant_worker()
compressor = config.get_compress_worker()
sr = config.get_sr_worker()
as_ = config.get_as_worker()
predict = config.get_predictor_worker()


def sudden_shutdown(*_, **__):
    global stat_mng
    stat_mng.cleanup()
    exit()


signal.signal(signal.SIGINT, sudden_shutdown)


def ptp1_prepare():
    global temp_path

    with torch.no_grad():
        for id_, (name, image) in enumerate(dataset):
            image = image.cpu()
            start_numpy = image.numpy()

            start_numpy = np.moveaxis(start_numpy, 0, 2)
            start_numpy = cv2.cvtColor(start_numpy, cv2.COLOR_RGB2BGR)
            if basic_size:
                if start_numpy.shape[::-1][1:] != basic_size:
                    start_numpy = cv2.resize(start_numpy, basic_size, interpolation=cv2.INTER_AREA)
            as_numpy = np.copy(start_numpy)
            image, as_prepare_time = as_.prepare_work(as_numpy)

            if vae:
                image, _ = vae.encode_work(image)
            if quant:
                (image, quant_params), _ = quant.quant_work(image)

            image, _ = compressor.compress_work(image)

            new_path = os.path.join(temp_path, name)
            with open(new_path, mode='wb') as wimage:
                wimage.write(image)


def ptp1_pipeline(n, intermediate_shape, dest_shape, dest_type, strict_sync, milliseconds_mode):
    global temp_path
    frame_queue = collections.deque(maxlen=(n+1))

    with torch.no_grad():
        for id_, (name, _) in enumerate(dataset):
            new_path = os.path.join(temp_path, name)
            with open(new_path, mode='rb') as rimage:
                image = rimage.read()

            torch.cuda.synchronize()
            # transmit_timepoint = time.time()

            image, decompress_time = compressor.decompress_work(image, intermediate_shape, dest_type,
                                                                strict_sync=strict_sync,
                                                                milliseconds_mode=milliseconds_mode)
            if quant:
                image, dequant_time = quant.dequant_work(image, strict_sync=strict_sync,
                                                         milliseconds_mode=milliseconds_mode)
            else:
                dequant_time = 0
            if vae:
                image, decoding_time = vae.decode_work(image, strict_sync=strict_sync,
                                                       milliseconds_mode=milliseconds_mode)
            else:
                decoding_time = 0

            end_numpy, as_restore_time = as_.restore_work(image, strict_sync=strict_sync,
                                                          milliseconds_mode=milliseconds_mode)

            end_numpy, superresolution_time = sr.sr_work(end_numpy, dest_size=dest_shape,
                                                         strict_sync=strict_sync,
                                                         milliseconds_mode=milliseconds_mode)

            # SR в эксперименте включает перевод данных, то есть AS:
            superresolution_time += as_restore_time
            as_restore_time = 0

            if predict and (n > 0):
                # Изначально некорректно, но время укажет
                frame_queue.append(end_numpy)
                images_to_predict = list(frame_queue)
                res_images, predictor_time = predict.predict_work(images_to_predict, n,
                                                                  strict_sync=strict_sync,
                                                                  milliseconds_mode=milliseconds_mode)
                frame_queue.extend(res_images)
            else:
                predictor_time = 0

            torch.cuda.synchronize()
            # end_time = time.time()
            # total_decoder_time = end_time - transmit_timepoint

            total_decoder_time = decompress_time + dequant_time + decoding_time + predictor_time
            # При правильном измерении примерно равны
            # total_time = end_time - beginning_time
            total_time = (n+1) * (1 + (999 * milliseconds_mode)) / total_decoder_time
            # (!!!) ВМЕСТО total_time -- FPS (!!!)

            stat_mng.write_stat([id_, name,
                                 0.0, 0.0, 0.0,
                                 0,
                                 0.0, 0.0, 0.0,
                                 0.0, as_restore_time,
                                 0.0, decoding_time,
                                 0.0, dequant_time,
                                 0.0, decompress_time,
                                 superresolution_time, predictor_time,
                                 0.0, total_decoder_time, total_time,
                                 0.0])

            if (id_ % progress_check) == 0:
                print("=== Прогресс: {}% ({}/{})".format(round(id_ / max_entries * 100, 3), id_, max_entries))
            if id_ >= max_entries:
                break


# Здесь исполнение пайплайна
# (!!!) ВМЕСТО total_time -- FPS (!!!)

global_dest_type = torch.float16
global_intermediate_shape = vae.z_shape
global_dest_shape = (1280, 720)
str_sync = True
ms_mode = True
num_of_predictions = 1

from_stat = "ptp1_statistics.csv"
from_summary = "ptp1_statistics_summary.csv"
from_dir = "ptp1_results"
os.makedirs(from_dir, exist_ok=True)
n_values = range(6)
shape_values = ((512, 512), (1280, 720), (1920, 1080))

ptp1_prepare()
for num_of_predictions in n_values:
    for global_dest_shape in shape_values:
        new_filename_base = "n{}_sh{}x{}_".format(num_of_predictions, *global_dest_shape)
        new_stat = os.path.join(from_dir, new_filename_base + from_stat)
        new_summary = os.path.join(from_dir, new_filename_base + from_summary)

        stat_mng = config.get_stat_mng()
        ptp1_pipeline(num_of_predictions, global_intermediate_shape,
                      global_dest_shape, global_dest_type,
                      str_sync, ms_mode)
        stat_mng.cleanup()

        os.rename(from_stat, new_stat)
        os.rename(from_summary, new_summary)

# Здесь исполнение пайплайна окончено
