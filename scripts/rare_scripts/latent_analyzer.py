import os
import sys
import csv
import math
import time

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
from utils.workers import (WorkerAutoencoderVQ_F4, WorkerAutoencoderVQ_F8, WorkerAutoencoderVQ_F16,
                           WorkerAutoencoderKL_F4, WorkerAutoencoderKL_F8, WorkerAutoencoderKL_F16,
                           WorkerAutoencoderKL_F32, WorkerAutoencoderInterface)


config_path = os.path.join("scripts", "rare_scripts", "latent_analyzer.ini")
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

# vae = config.get_autoencoder_worker()
quant = config.get_quant_worker()
quant.unlock()
as_ = config.get_as_worker()

p_var = 0.95
n = 1000
student = scipy.stats.t.ppf((1+p_var)/2, n)

all_vae = [
    WorkerAutoencoderVQ_F4(config_path="dependence/config/vq-f4.yaml", ckpt_path="dependence/ckpt/vq-f4.ckpt"),
    WorkerAutoencoderVQ_F8(config_path="dependence/config/vq-f8.yaml", ckpt_path="dependence/ckpt/vq-f8.ckpt"),
    WorkerAutoencoderVQ_F16(config_path="dependence/config/vq-f16.yaml", ckpt_path="dependence/ckpt/vq-f16.ckpt"),
    WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml", ckpt_path="dependence/ckpt/kl-f4.ckpt"),
    WorkerAutoencoderKL_F8(config_path="dependence/config/kl-f8.yaml", ckpt_path="dependence/ckpt/kl-f8.ckpt"),
    WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml", ckpt_path="dependence/ckpt/kl-f16.ckpt"),
    WorkerAutoencoderKL_F32(config_path="dependence/config/kl-f32.yaml", ckpt_path="dependence/ckpt/kl-f32.ckpt"),
]

all_list = ["WorkerAutoencoderVQ_F4", "WorkerAutoencoderVQ_F8", "WorkerAutoencoderVQ_F16",
            "WorkerAutoencoderKL_F4", "WorkerAutoencoderKL_F8", "WorkerAutoencoderKL_F16",
            "WorkerAutoencoderKL_F32"
]

def latent_pipeline(vae: WorkerAutoencoderInterface):
    global all_values
    global sigma
    global means

    dataset = config.get_dataset()
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
            image, as_prepare_time = as_.prepare_work(as_numpy, dest_type=vae.nominal_type,
                                                      strict_sync=True, milliseconds_mode=True)

            image, _ = vae.encode_work(image, strict_sync=True, milliseconds_mode=True)
            # (image, quant_params), _ = quant.quant_work(image)

            # all_params.append(quant_params)

            # TEMP
            # time.sleep(1)
            image = torch.flatten(image)
            all_values = np.concatenate([all_values, image.cpu().numpy()])
            if id_ > 9:
                break


# Здесь исполнение пайплайна


one_width = 500
one_height = 500
temp_file = open("temp.txt", mode='w')
for vae, vae_name in zip(all_vae, all_list):
    # print("===", vae_name, "===", file=temp_file)
    print("===", vae_name, "===")
    print("!!!")

    # all_params = []
    all_values = np.array([])

    latent_pipeline(vae)
    if vae_name == "WorkerAutoencoderKL_F4":
        band = 2
    elif vae_name == "WorkerAutoencoderKL_F8":
        band = 3
    elif vae_name == "WorkerAutoencoderKL_F16":
        band = 2
    elif vae_name == "WorkerAutoencoderKL_F32":
        band = "silverman"
    else:
        band = "silverman"

    kde = KernelDensity(kernel="gaussian", bandwidth=band)
    all_values = all_values.reshape(-1, 1)
    kde.fit(all_values)
    step = 0.01
    esers = np.arange(all_values.min(), all_values.max() + step, step)
    esers_ = esers.reshape(-1, 1)
    yers = np.exp(kde.score_samples(esers_))*100

    print(scipy.integrate.simpson(yers, x=esers))

    plt.xlabel("Значение латентого пространства")
    plt.ylabel("Плотность вероятности, %")
    plt.plot(esers, yers)
    plt.show()


    """
    print(all_params[:10])

    param_num = len(all_params[0])
    for i in range(param_num):
        print("- Параметр", i, file=temp_file)
        paramer = np.array([k[i] for k in all_params])
        paramer_m = paramer.mean().item()
        paramer_std = paramer.std().item()
        paramer_delta = student * paramer_std / math.sqrt(n)
        print(paramer_m, file=temp_file)
        print(paramer_delta, file=temp_file)
    """

    #print("Mean={}±{}".format(means_m, means_delta), file=temp_file)
    #print("Sigma={}±{}".format(sigma_m, sigma_delta), file=temp_file)

    print()
    print()
temp_file.close()

"""
one_width = 500
one_height = 500
temp_file = open("temp.txt", mode='w')
for vae, vae_name in zip(all_vae, all_list):
    print("===", vae_name, "===", file=temp_file)
    print("!!!")

    chartfile = "{}.html".format(vae_name)

    means = []
    sigma = []
    all_values = np.array([], dtype=np.float16)

    latent_pipeline(vae)

    means = np.array(means)
    sigma = np.array(sigma)
    means_m = means.mean().item()
    sigma_m = sigma.mean().item()
    means_std = means.std().item()
    sigma_std = sigma.std().item()
    means_delta = student * means_std / math.sqrt(n)
    sigma_delta = student * sigma_std / math.sqrt(n)

    shapiro = scipy.stats.shapiro(all_values)

    print("Mean={}±{}".format(means_m, means_delta), file=temp_file)
    print("Sigma={}±{}".format(sigma_m, sigma_delta), file=temp_file)
    print("Shapiro_pval={}".format(shapiro.pvalue), file=temp_file)

    all_values = np.random.choice(all_values, 1_000_000)
    dataset = pd.DataFrame(data={"X": all_values})
    chart = alt.Chart(dataset).transform_joinaggregate(
        total='count(*)'
    ).transform_calculate(
        pct='1 / datum.total'
    ).properties(width=one_width, height=one_height)
    chart = chart.mark_bar()  # cornerRadius=5
    chart = chart.encode(
        x=alt.X("X", title="Значение латентого пространства", bin=True),
        y=alt.Y("sum(pct):Q", title="Процент значений, %", axis=alt.Axis(format='%')),
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=18
    )
    chart.save(chartfile)

    print()
    print()
temp_file.close()
"""
