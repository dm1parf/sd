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
import torchvision


def pipeline(datadir, indexes):
    config_path = os.path.join("scripts", "rare_scripts", "latent_analyzer2.ini")
    config = ConfigManager(config_path)

    quant = config.get_quant_worker()
    as_ = config.get_as_worker()
    vae = config.get_autoencoder_worker()
    compressor = config.get_compress_worker()

    lsize = reduce(lambda a, b: a*b, vae.z_shape)
    basic_size = config.get_basic_size()

    cr_data = []

    with torch.no_grad():
        for indexer in indexes:
            this_file = os.path.join(datadir, "{}.jpg".format(str(indexer)))
            image = torchvision.io.read_image(this_file, mode=torchvision.io.ImageReadMode.RGB)

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
            cr = 100 * (1 - msize / lsize)
            cr_data.append(cr)

    return cr_data


direr = "dataset_preparation/sakhalin_dataset_1000seq"
seq1 = list(range(1000))
seq2 = list(range(500, 1000)) + list(range(0, 500))

if os.path.isfile("seq1.npy"):
    y1 = np.load("seq1.npy")
else:
    y1 = pipeline(direr, seq1)
    y1 = np.array(y1)
    np.save("seq1.npy", y1)
if os.path.isfile("seq2.npy"):
    y2 = np.load("seq2.npy")
else:
    y2 = pipeline(direr, seq2)
    y2 = np.array(y2)
    np.save("seq2.npy", y2)

source = pd.DataFrame({
    'seq1': [str(i) for i in seq1],
    'seq2': [str(i) for i in seq2],
    'y1': y1,
    'y2': y2
})
print(seq1)
print(seq2)

heighter = 100
widther = 1000
high = alt.Chart(source).mark_bar().encode(
    x=alt.X("seq1", title="Номер кадра", sort=None),
    y=alt.Y("y1", title="CR, %")
).properties(height=heighter,
             width=widther)

low = alt.Chart(source).mark_bar().encode(
    x=alt.X("seq2", title="Номер кадра", sort=None),
    y=alt.Y("y2", title="CR, %")
).properties(height=heighter,
             width=widther)
result = alt.vconcat(high, low)
result.save("seqs.html")
