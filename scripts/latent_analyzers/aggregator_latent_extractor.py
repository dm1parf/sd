import os
import sys
import csv
import signal
import cv2
import torch
import numpy as np
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager


config_path = os.path.join("scripts", "rare_scripts", "latent_extractor.ini")
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
# compressor = config.get_compress_worker()
# sr = config.get_sr_worker()
as_ = config.get_as_worker()
# predict = config.get_predictor_worker()

experiment_settings = config.config["ExperimentSettings"]
statistics_dir = experiment_settings["stat_dir"]
os.makedirs(statistics_dir, exist_ok=True)
statistics_filename = os.path.join(statistics_dir, experiment_settings["stat_filename"])
statistics_filename = statistics_filename.format("all_latent")

base_rows = ["n"]
for i in range(1, 9):
    for j in range(1, 33):
        for k in range(1, 33):
            new_row = "{}_{}_{}".format(i, j, k)
            base_rows.append(new_row)

filer = open(statistics_filename, mode='w', encoding='utf-8', newline='\n')
csvw = csv.writer(filer, delimiter=',')
csvw.writerow(base_rows)


def sudden_shutdown(*_, **__):
    filer.flush()
    filer.close()
    exit()


signal.signal(signal.SIGINT, sudden_shutdown)


def write_latent_to_file(frame_id, latent):
    global base_rows
    global csvw

    entry = [frame_id + 1]

    for i in range(1, 9):
        for j in range(1, 33):
            for k in range(1, 33):
                this_val = latent[0, i-1, j-1, k-1]
                this_val = round(this_val.item(), 4)
                entry.append(this_val)
    csvw.writerow(entry)

def latent_pipeline():
    global temp_path

    with torch.no_grad():
        for id_, (name, image) in enumerate(dataset):
            print(id_)
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

            write_latent_to_file(id_, image)


# Здесь исполнение пайплайна

latent_pipeline()
sudden_shutdown()