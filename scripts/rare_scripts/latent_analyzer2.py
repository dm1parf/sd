import os
import sys
import csv
import math
import time
from functools import reduce

import cv2
import torch
import numpy as np
import pandas as pd
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager
import altair as alt
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KernelDensity
from utils.workers import WorkerAutoencoderVQ_F8


#datafile = "arc.npy"
if True:
    config_path = os.path.join("scripts", "rare_scripts", "latent_analyzer2.ini")
    config = ConfigManager(config_path)

    dataset = config.get_dataset()
    basic_size = config.get_basic_size()
    dataset_len = len(dataset)
    print("Dataset length:", dataset_len)

    progress_check = config.get_progress_check()

    max_entries = config.get_max_entries()
    if max_entries == 0:
        max_entries = dataset_len
    else:
        max_entries = min(max_entries, dataset_len)

    # vae = config.get_autoencoder_worker()
    quant = config.get_quant_worker()
    as_ = config.get_as_worker()
    vae = config.get_autoencoder_worker()
    compressor = config.get_compress_worker()

    lsize = reduce(lambda a, b: a*b, vae.z_shape)
    cr_data = []
    msize_data = []

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
            image, as_prepare_time = as_.prepare_work(as_numpy, dest_type=vae.nominal_type)

            image, _ = vae.encode_work(image)
            (image, quant_params), _ = quant.quant_work(image)
            binary, _ = compressor.compress_work(image)

            msize = len(binary)
            msize_data.append(msize)
            cr = 100 * (1 - msize / lsize)
            cr_data.append(cr)
            
            if (id_ % 1000) == 0:
                print(id_, "/", dataset_len)

    cr_data = np.array(cr_data)
    msize_data = np.array(msize_data)
    # np.save(datafile, cr_data)

# cr_data

print("--!!!--")
print("--- CR ---")
print("mean:", cr_data.mean())
print("std:", cr_data.std())
print("--- MSize ---")
print("mean:", msize_data.mean())
print("std:", msize_data.std())

"""
band = 0.5
kde = KernelDensity(kernel="gaussian", bandwidth=band)
all_values = cr_data.reshape(-1, 1)
kde.fit(all_values)
step = 0.01
esers = np.arange(all_values.min(), all_values.max() + step, step)
esers_ = esers.reshape(-1, 1)
yers = np.exp(kde.score_samples(esers_)) * 100

print(scipy.integrate.simpson(yers, x=esers))

plt.xlabel("CR, %")
plt.ylabel("Плотность вероятности, %")
ax = plt.gca()
ax.set_xlim([8.2, 20.0])
plt.plot(esers, yers)
plt.show()
"""