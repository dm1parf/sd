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
import re
import math
from skimage.metrics import structural_similarity
from utils.workers import WorkerPredictorDMVFN
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt

preder = True
recurrer = True


arguments = argparse.ArgumentParser(prog="Исследование предсказания будущего",
                                    description="Сделано для исследования предсказания будущего.")
arguments.add_argument('--overwrite', dest="overwrite", action='store_true', default=False)
arguments.add_argument('--demonstrate', dest="demonstrate", action='store_true', default=False)
arguments.add_argument("--every", dest="every", type=int, default=1000, help="Каждые столько мс демонстрировать")
args = arguments.parse_args()
dem_every = args.every
demonstrate = args.demonstrate
overwrite = args.overwrite


datafile_format = "scripts/rare_scripts/predict_future_data_{}.csv"
datafile = "scripts/rare_scripts/predict_future_data_40.csv"
statrec = "scripts/rare_scripts/predict_future_data_vid.csv"


def mse_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики MSE.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = np.mean((image1 - image2) ** 2)

    return mse


def ssim_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики SSIM.
    На вход подаются две картинки в формате cv2 (numpy)."""

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

    return score


def psnr_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики PSNR.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


print("File not found. Starting experiments...")

FPS = 25
d = list(range(1, 26))
dt = 1000 // FPS
dT = [dt * i for i in d]
B = [i + 1 for i in d]


if os.path.isfile(datafile) and not demonstrate and not overwrite:
    print("File found! Loading...")

    ssim_means = []
    mse_means = []
    psnr_means = []
    for this_d, this_B, this_dT in zip(d, B, dT):
        # print(f"=== d = {this_d}, B = {this_B}, dT = {this_dT} ===")
        this_datafile = datafile_format.format(this_dT)

        dataset = pd.read_csv(this_datafile)
        ssim_data = dataset["ssim"].to_numpy()
        mse_data = dataset["mse"].to_numpy()
        psnr_data = dataset["psnr"].to_numpy()

        ssim_mean = ssim_data.mean()
        ssim_sigma = ssim_data.std()
        mse_mean = mse_data.mean()
        mse_sigma = mse_data.std()
        psnr_mean = psnr_data.mean()
        psnr_sigma = psnr_data.std()

        ssim_means.append(ssim_mean)
        mse_means.append(mse_mean)
        psnr_means.append(psnr_mean)

        print(f"{this_d} & {this_dT} & {ssim_mean:.2f} & {ssim_sigma:.2f} & {mse_mean:.2f} & {mse_sigma:.2f} & {psnr_mean:.2f} & {psnr_sigma:.2f} \\\\")

    print("=== График SSIM ===")
    plt.xlabel("dT")
    plt.ylabel("SSIMср")
    plt.ylim([0.0, 1.0])
    plt.plot(dT, ssim_means)
    plt.show()

    print("=== График MSE ===")
    plt.xlabel("dT")
    plt.ylabel("MSEср")
    plt.plot(dT, mse_means)
    plt.show()

    print("=== График PSNR ===")
    plt.xlabel("dT")
    plt.ylabel("PSNRср")
    plt.plot(dT, psnr_means)
    plt.show()

    dataset_40 = pd.read_csv(datafile)
    dataset_rec = pd.read_csv(statrec)

    ti_series_1 = dataset_40["t_i"].to_numpy()
    ti_series_2 = dataset_rec["t_i"].to_numpy()

    ssim_predict = dataset_40["ssim"].to_numpy()
    ssim_real = dataset_rec["ssim"].to_numpy()
    print(ssim_predict.shape, ssim_real.shape)

    mse_predict = dataset_40["mse"].to_numpy()
    mse_real = dataset_rec["mse"].to_numpy()

    psnr_predict = dataset_40["psnr"].to_numpy()
    psnr_real = dataset_rec["psnr"].to_numpy()

    print("=== График сравнения SSIM ===")
    ssim_corr = np.corrcoef(ssim_real, ssim_predict)
    print("Correl:", ssim_corr)
    #plt.xlabel("ti")
    #plt.ylabel("SSIM")
    #plt.plot(ti_series_2, ssim_real, color="blue", label="Реальные кадры")
    #plt.plot(ti_series_1, ssim_predict, color="red", label="Предсказанные кадры")
    #plt.legend(loc='upper left')
    plt.xlabel("SSIM реальных кадров")
    plt.ylabel("SSIM предсказанных кадров")
    plt.scatter(ssim_real, ssim_predict)
    plt.show()

    print("=== График сравнения MSE ===")
    mse_corr = np.corrcoef(mse_real, mse_predict)
    print("Correl:", mse_corr)
    #plt.xlabel("ti")
    #plt.ylabel("MSE")
    #plt.plot(ti_series_2, mse_real, color="blue", label="Реальные кадры")
    #plt.plot(ti_series_1, mse_predict, color="red", label="Предсказанные кадры")
    #plt.legend(loc='upper left')
    plt.xlabel("MSE реальных кадров")
    plt.ylabel("MSE предсказанных кадров")
    plt.scatter(mse_real, mse_predict)
    plt.show()

    print("=== График сравнения PSNR ===")
    psnr_corr = np.corrcoef(psnr_real, psnr_predict)
    print("Correl:", psnr_corr)
    #plt.xlabel("ti")
    #plt.ylabel("PSNR")
    #plt.plot(ti_series_2, psnr_real, color="blue", label="Реальные кадры")
    #plt.plot(ti_series_1, psnr_predict, color="red", label="Предсказанные кадры")
    #plt.legend(loc='upper left')
    plt.xlabel("PSNR реальных кадров")
    plt.ylabel("PSNR предсказанных кадров")
    plt.scatter(psnr_real, psnr_predict)
    plt.show()

    sys.exit()


dataset = "dataset_preparation/spbsut_25fps_full"
dataset_dict = dict()
all_images = os.listdir(dataset)
factor = r"(\d{1,5})\..*"
for img_file in all_images:
    full_filename = os.path.join(dataset, img_file)
    try:
        res = re.search(factor, img_file)
        timer = dt * int(res[1])
    except:
        continue
    dataset_dict[timer] = full_filename

predictor = WorkerPredictorDMVFN(path="dependence/config/dmvfn_city.pkl")

times = sorted(list(dataset_dict.keys()))
if preder:
    for this_d, this_B, this_dT in zip(d, B, dT):
        print(f"=== d = {this_d}, B = {this_B}, dT = {this_dT} ===")

        this_datafile = datafile_format.format(this_dT)
        if not demonstrate:
            this_dater = open(this_datafile, mode='w', newline='')
            this_csv = csv.writer(this_dater)
            this_csv.writerow(["d", "B", "dT", "t_i", "t_ipd", "ssim", "mse", "psnr"])

        prediction_buffer = []
        for ti in times:
            img_file = dataset_dict[ti]
            frame = cv2.imread(img_file)

            prediction_buffer.append(frame)
            if len(prediction_buffer) == this_B:
                actual_buffer = [prediction_buffer[0], prediction_buffer[-1]]
                try:
                    predict_frames, _ = predictor.predict_work(actual_buffer, 1)
                except:
                    print("!!!", ti)
                    print([i.shape for i in actual_buffer])
                    continue
                predict_frame = predict_frames[0]

                t_ipd = ti + this_dT
                if t_ipd in dataset_dict:
                    real_frame = cv2.imread(dataset_dict[t_ipd])
                    ssim = ssim_metric(predict_frame, real_frame)
                    mse = mse_metric(predict_frame, real_frame)
                    psnr = psnr_metric(predict_frame, real_frame)

                    if demonstrate:
                        print(t_ipd, ":", ssim, mse, psnr)
                        if (t_ipd % dem_every) == 0:
                            cv2.imshow("===", predict_frame)
                            cv2.waitKey(1000)
                    else:
                        stat_data = [this_d, this_B, this_dT, ti, t_ipd, ssim, mse, psnr]
                        this_csv.writerow(stat_data)

                prediction_buffer.pop(0)
        if not demonstrate:
            this_dater.flush()
            this_dater.close()

times = sorted(list(dataset_dict.keys()))
if recurrer:
    with open(statrec, mode='w', newline='') as srec:
        csvrec = csv.writer(srec)
        csvrec.writerow(["d", "t_i", "ssim", "mse", "psnr"])

        ti = 0
        previous_frame = None
        while ti in times:
            this_fn = dataset_dict[ti]
            frame = cv2.imread(this_fn)
            if previous_frame is not None:
                ssim = ssim_metric(previous_frame, frame)
                mse = mse_metric(previous_frame, frame)
                psnr = psnr_metric(previous_frame, frame)
                d = ti // dt

                this_data = [d, ti, ssim, mse, psnr]
                csvrec.writerow(this_data)
            previous_frame = frame

            ti += dt

