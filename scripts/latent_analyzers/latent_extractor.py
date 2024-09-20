import os
import sys
import csv
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

base_rows = ["1_A_B_C", "1", "2", "3", "4", "5", "6", "7", "8"]


def write_latent_to_file(frame_id, latent):
    global base_rows
    global statistics_filename

    this_filename = statistics_filename.format(frame_id)
    with open(this_filename, mode='w', encoding='utf-8', newline='\n') as wf:
        csvw = csv.writer(wf, delimiter=',')
        csvw.writerow(base_rows)

        for i in range(32):
            for j in range(32):
                this_row = "{}_{}".format(i+1, j+1)
                this_values = [this_row] + [round(k, 4) for k in latent[0, :, i, j].tolist()]
                csvw.writerow(this_values)

def latent_pipeline():
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

            write_latent_to_file(id_, image)


# Здесь исполнение пайплайна

latent_pipeline()
