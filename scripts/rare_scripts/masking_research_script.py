import os
import sys

import torch

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import matplotlib.pyplot as plt
from models.masking.masker import MaskMaster
from models.masking.bsrnn import BSRNN
from models.masking.simple_mask_nn import SBSRNN_M, SBSRNN_F
from models.masking.regression import LG_RPM, LN_RPM
from utils.workers import WorkerASDummy, WorkerQuantLinear, WorkerAutoencoderKL_F16, WorkerCompressorDummy
from scripts.fpv_ctvp_emulators.mock_fpv_ctvp_decoder import NeuroCodec
from skimage.metrics import structural_similarity
import pandas
import numpy as np
import cv2
import csv
import time
import math


data_image = False
test_bsrnn = True
test_sbsrnn_m = True
test_sbsrnn_f = True

test_lg_rpm = True
test_ln_rpm = True
test_summary = True


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


if data_image:
    x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rmse_labels = [0.000529, 0.000410, 0.000295, 0.000189, 0.000108, 0.000079, 0.000076, 0.000076, 0.000076, 0.000076]

    plt.xlabel("Эпоха")
    plt.ylabel("RMSEср")
    plt.plot(x_labels, rmse_labels)
    plt.axvline(x=6, color='r', linestyle='dashed')
    plt.text(6, 0.05, "7", color='red')
    plt.savefig(r'D:\UserData\Работа\Проекты_статей\Маскирование\Маскирование2_рисунки\3.jpg', dpi=300)
    # plt.show()
    plt.close()


dataset = "dataset_preparation/spbsut_dataset_1000"
weight_mask = "models/masking/bsrnn_weights/bsrnn_{}_{}"
# experiment_result = "scripts/rare_scripts/masking_result"
experiment_result = r"D:\UserData\Работа\Проекты_статей\Маскирование\masking_result"
all_p = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
         0.95]
length = 16_384
typer = torch.float16
device = "cuda"
os.makedirs(experiment_result, exist_ok=True)


if test_bsrnn:
    res_file = "bsrnn_{:.2f}.csv"
    img1_file = "bsrnn_{:.2f}_img1.jpg"
    img2_file = "bsrnn_{:.2f}_img2.jpg"
    weights_path = "models/masking/bsrnn_weights/bsrnn_{}_{:.2f}"
    res_fullpath = os.path.join(experiment_result, res_file)
    img1_fullpath = os.path.join(experiment_result, img1_file)
    img2_fullpath = os.path.join(experiment_result, img2_file)

    bsrnn_ssim = []
    bsrnn_mse = []
    bsrnn_psnr = []
    bsrnn_mask_time = []
    bsrnn_demask_time = []
    bsrnn_black_rate = []

    for p in all_p:
        resp_fullpath = res_fullpath.format(p)
        imgone_fullpath = img1_fullpath.format(p)
        imgtwo_fullpath = img2_fullpath.format(p)
        weight_fullpath = weights_path.format(length, p)

        # Эксперименты

        if not os.path.isfile(resp_fullpath):
            print("=== Испытание BSRNN: p = {} ===".format(p))
            model = BSRNN(length=length, p=p)
            model = model.to(dtype=typer, device=device)
            model.load_state_dict(torch.load(weight_fullpath))  # weights_only=True strict=False
            masker = MaskMaster(length=length, p=p)
            as_ = WorkerASDummy()
            kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                             ckpt_path="dependence/ckpt/kl-f16.ckpt")
            quant = WorkerQuantLinear()
            quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
            compress = WorkerCompressorDummy()
            codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

            img_flag = True

            with open(resp_fullpath, mode='w', newline='') as wf:
                csv_wr = csv.writer(wf)
                csv_wr.writerow(["id", "ssim", "mse", "psnr", "is_black", "mask_time", "demask_time"])
                id_ = 0

                for root, dirs, files in os.walk(dataset):
                    for file in files:
                        from_filepath = os.path.join(root, file)
                        image = cv2.imread(from_filepath)
                        latent_bytes = codec.encode_frame(image)

                        pre_latent = torch.frombuffer(latent_bytes, dtype=torch.uint8)
                        pre_latent = pre_latent.to(device=device)
                        pre_latent = pre_latent.reshape(1, -1)
                        latent = masker.prepare_latent(pre_latent, typer=typer)
                        torch.cuda.synchronize()
                        a = time.time()
                        mini_latent = masker.get_latent(latent)
                        torch.cuda.synchronize()
                        b = time.time()
                        mask = model(mini_latent)
                        pre_restore_latent = masker.recompose(mini_latent, mask, batch_size=1)
                        restore_latent = masker.restore_latent(pre_restore_latent)
                        torch.cuda.synchronize()
                        c = time.time()

                        restore_latent = restore_latent.to('cpu')
                        numpy_img = restore_latent.numpy()
                        byter = numpy_img.tobytes()
                        res_image = codec.decode_frame(byter)

                        mask_time = round((b - a) * 1000, 2)  # мс
                        demask_time = round((c - b) * 1000, 2)  # мс

                        if img_flag:
                            cv2.imwrite(imgone_fullpath, image)
                            cv2.imwrite(imgtwo_fullpath, res_image)
                            img_flag = False

                        ssim = ssim_metric(image, res_image)
                        mse = mse_metric(image, res_image)
                        psnr = psnr_metric(image, res_image)
                        ssim = round(ssim, 2)
                        mse = round(mse, 2)
                        psnr = round(psnr, 2)

                        if res_image.any():
                            is_black = 0
                        else:
                            is_black = 1

                        csv_wr.writerow([id_, ssim, mse, psnr, is_black, mask_time, demask_time])
                        id_ += 1

        # Обработки

        bsrnn_data = pandas.read_csv(resp_fullpath)
        ssim_data = bsrnn_data["ssim"].to_numpy()
        mse_data = bsrnn_data["mse"].to_numpy()
        psnr_data = bsrnn_data["psnr"].to_numpy()
        masktime_data = bsrnn_data["mask_time"].to_numpy()
        demasktime_data = bsrnn_data["demask_time"].to_numpy()
        blackrate_data = bsrnn_data["is_black"].to_numpy()

        ssim = round(ssim_data.mean(), 2)
        mse = round(mse_data.mean(), 2)
        psnr = round(psnr_data.mean(), 2)
        mask_time = round(masktime_data.mean(), 2)
        demask_time = round(demasktime_data.mean(), 2)
        black_rate = round(len(blackrate_data[blackrate_data == 1])/len(blackrate_data) * 100, 2)

        bsrnn_ssim.append(ssim)
        bsrnn_mse.append(mse)
        bsrnn_psnr.append(psnr)
        bsrnn_mask_time.append(mask_time)
        bsrnn_demask_time.append(demask_time)
        bsrnn_black_rate.append(black_rate)

if test_sbsrnn_m:
    res_file = "sbsrnn_m_{:.2f}.csv"
    img1_file = "sbsrnn_m_{:.2f}_img1.jpg"
    img2_file = "sbsrnn_m_{:.2f}_img2.jpg"
    weights_path = "models/masking/sbsrnn_m_weights/sbsrnn_m_{}_{:.2f}"
    res_fullpath = os.path.join(experiment_result, res_file)
    img1_fullpath = os.path.join(experiment_result, img1_file)
    img2_fullpath = os.path.join(experiment_result, img2_file)

    sbsrnn_m_ssim = []
    sbsrnn_m_mse = []
    sbsrnn_m_psnr = []
    sbsrnn_m_mask_time = []
    sbsrnn_m_demask_time = []
    sbsrnn_m_black_rate = []

    for p in all_p:
        resp_fullpath = res_fullpath.format(p)
        imgone_fullpath = img1_fullpath.format(p)
        imgtwo_fullpath = img2_fullpath.format(p)
        weight_fullpath = weights_path.format(length, p)

        # Эксперименты

        if not os.path.isfile(resp_fullpath):
            print("=== Испытание SBSRNN-M: p = {} ===".format(p))
            model = SBSRNN_M(length=length, p=p)
            model.load_state_dict(torch.load(weight_fullpath))  # weights_only=True strict=False
            model = model.to(dtype=typer, device=device)
            masker = MaskMaster(length=length, p=p)
            as_ = WorkerASDummy()
            kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                             ckpt_path="dependence/ckpt/kl-f16.ckpt")
            quant = WorkerQuantLinear()
            quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
            compress = WorkerCompressorDummy()
            codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

            img_flag = True

            with open(resp_fullpath, mode='w', newline='') as wf:
                csv_wr = csv.writer(wf)
                csv_wr.writerow(["id", "ssim", "mse", "psnr", "is_black", "mask_time", "demask_time"])
                id_ = 0

                for root, dirs, files in os.walk(dataset):
                    for file in files:
                        from_filepath = os.path.join(root, file)
                        image = cv2.imread(from_filepath)
                        latent_bytes = codec.encode_frame(image)

                        pre_latent = torch.frombuffer(latent_bytes, dtype=torch.uint8)
                        pre_latent = pre_latent.to(device=device)
                        pre_latent = pre_latent.reshape(1, -1)
                        latent = masker.prepare_latent(pre_latent, typer=typer)
                        torch.cuda.synchronize()
                        a = time.time()
                        mini_latent = masker.get_latent(latent)
                        torch.cuda.synchronize()
                        b = time.time()
                        mask = model(mini_latent)
                        pre_restore_latent = masker.recompose(mini_latent, mask, batch_size=1)
                        restore_latent = masker.restore_latent(pre_restore_latent)
                        torch.cuda.synchronize()
                        c = time.time()

                        restore_latent = restore_latent.to('cpu')
                        numpy_img = restore_latent.numpy()
                        byter = numpy_img.tobytes()
                        res_image = codec.decode_frame(byter)

                        mask_time = round((b - a) * 1000, 2)  # мс
                        demask_time = round((c - b) * 1000, 2)  # мс

                        if img_flag:
                            cv2.imwrite(imgone_fullpath, image)
                            cv2.imwrite(imgtwo_fullpath, res_image)
                            img_flag = False

                        ssim = ssim_metric(image, res_image)
                        mse = mse_metric(image, res_image)
                        psnr = psnr_metric(image, res_image)
                        ssim = round(ssim, 2)
                        mse = round(mse, 2)
                        psnr = round(psnr, 2)

                        if res_image.any():
                            is_black = 0
                        else:
                            is_black = 1

                        csv_wr.writerow([id_, ssim, mse, psnr, is_black, mask_time, demask_time])
                        id_ += 1

        # Обработки

        sbsrnn_m_data = pandas.read_csv(resp_fullpath)
        ssim_data = sbsrnn_m_data["ssim"].to_numpy()
        mse_data = sbsrnn_m_data["mse"].to_numpy()
        psnr_data = sbsrnn_m_data["psnr"].to_numpy()
        masktime_data = sbsrnn_m_data["mask_time"].to_numpy()
        demasktime_data = sbsrnn_m_data["demask_time"].to_numpy()
        blackrate_data = sbsrnn_m_data["is_black"].to_numpy()

        ssim = round(ssim_data.mean(), 2)
        mse = round(mse_data.mean(), 2)
        psnr = round(psnr_data.mean(), 2)
        mask_time = round(masktime_data.mean(), 2)
        demask_time = round(demasktime_data.mean(), 2)
        black_rate = round(len(blackrate_data[blackrate_data == 1])/len(blackrate_data) * 100, 2)

        sbsrnn_m_ssim.append(ssim)
        sbsrnn_m_mse.append(mse)
        sbsrnn_m_psnr.append(psnr)
        sbsrnn_m_mask_time.append(mask_time)
        sbsrnn_m_demask_time.append(demask_time)
        sbsrnn_m_black_rate.append(black_rate)

if test_sbsrnn_f:
    res_file = "sbsrnn_f_{:.2f}.csv"
    img1_file = "sbsrnn_f_{:.2f}_img1.jpg"
    img2_file = "sbsrnn_f_{:.2f}_img2.jpg"
    weights_path = "models/masking/sbsrnn_f_weights/sbsrnn_f_{}_{:.2f}"
    res_fullpath = os.path.join(experiment_result, res_file)
    img1_fullpath = os.path.join(experiment_result, img1_file)
    img2_fullpath = os.path.join(experiment_result, img2_file)

    sbsrnn_f_ssim = []
    sbsrnn_f_mse = []
    sbsrnn_f_psnr = []
    sbsrnn_f_mask_time = []
    sbsrnn_f_demask_time = []
    sbsrnn_f_black_rate = []

    for p in all_p:
        resp_fullpath = res_fullpath.format(p)
        imgone_fullpath = img1_fullpath.format(p)
        imgtwo_fullpath = img2_fullpath.format(p)
        weight_fullpath = weights_path.format(length, p)

        # Эксперименты

        if not os.path.isfile(resp_fullpath):
            print("=== Испытание SBSRNN-F: p = {} ===".format(p))
            model = SBSRNN_F(length=length, p=p)
            model.load_state_dict(torch.load(weight_fullpath))  # weights_only=True strict=False
            model = model.to(dtype=typer, device=device)
            masker = MaskMaster(length=length, p=p)
            as_ = WorkerASDummy()
            kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                             ckpt_path="dependence/ckpt/kl-f16.ckpt")
            quant = WorkerQuantLinear()
            quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
            compress = WorkerCompressorDummy()
            codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

            img_flag = True

            with open(resp_fullpath, mode='w', newline='') as wf:
                csv_wr = csv.writer(wf)
                csv_wr.writerow(["id", "ssim", "mse", "psnr", "is_black", "mask_time", "demask_time"])
                id_ = 0

                for root, dirs, files in os.walk(dataset):
                    for file in files:
                        from_filepath = os.path.join(root, file)
                        image = cv2.imread(from_filepath)
                        latent_bytes = codec.encode_frame(image)

                        pre_latent = torch.frombuffer(latent_bytes, dtype=torch.uint8)
                        pre_latent = pre_latent.to(device=device)
                        pre_latent = pre_latent.reshape(1, -1)
                        latent = masker.prepare_latent(pre_latent, typer=typer)
                        torch.cuda.synchronize()
                        a = time.time()
                        mini_latent = masker.get_latent(latent)
                        torch.cuda.synchronize()
                        b = time.time()
                        restore_latent = model(mini_latent)
                        # pre_restore_latent = masker.recompose(mini_latent, mask, batch_size=1)
                        restore_latent = masker.restore_latent(restore_latent)
                        torch.cuda.synchronize()
                        c = time.time()

                        restore_latent = restore_latent.to('cpu')
                        numpy_img = restore_latent.numpy()
                        byter = numpy_img.tobytes()
                        res_image = codec.decode_frame(byter)

                        mask_time = round((b - a) * 1000, 2)  # мс
                        demask_time = round((c - b) * 1000, 2)  # мс

                        if img_flag:
                            cv2.imwrite(imgone_fullpath, image)
                            cv2.imwrite(imgtwo_fullpath, res_image)
                            img_flag = False

                        ssim = ssim_metric(image, res_image)
                        mse = mse_metric(image, res_image)
                        psnr = psnr_metric(image, res_image)
                        ssim = round(ssim, 2)
                        mse = round(mse, 2)
                        psnr = round(psnr, 2)

                        if res_image.any():
                            is_black = 0
                        else:
                            is_black = 1

                        csv_wr.writerow([id_, ssim, mse, psnr, is_black, mask_time, demask_time])
                        id_ += 1

        # Обработки

        sbsrnn_f_data = pandas.read_csv(resp_fullpath)
        ssim_data = sbsrnn_f_data["ssim"].to_numpy()
        mse_data = sbsrnn_f_data["mse"].to_numpy()
        psnr_data = sbsrnn_f_data["psnr"].to_numpy()
        masktime_data = sbsrnn_f_data["mask_time"].to_numpy()
        demasktime_data = sbsrnn_f_data["demask_time"].to_numpy()
        blackrate_data = sbsrnn_f_data["is_black"].to_numpy()

        ssim = round(ssim_data.mean(), 2)
        mse = round(mse_data.mean(), 2)
        psnr = round(psnr_data.mean(), 2)
        mask_time = round(masktime_data.mean(), 2)
        demask_time = round(demasktime_data.mean(), 2)
        black_rate = round(len(blackrate_data[blackrate_data == 1])/len(blackrate_data) * 100, 2)

        sbsrnn_f_ssim.append(ssim)
        sbsrnn_f_mse.append(mse)
        sbsrnn_f_psnr.append(psnr)
        sbsrnn_f_mask_time.append(mask_time)
        sbsrnn_f_demask_time.append(demask_time)
        sbsrnn_f_black_rate.append(black_rate)

if test_lg_rpm:
    res_file = "lg_rpm_{:.2f}.csv"
    img1_file = "lg_rpm_{:.2f}_img1.jpg"
    img2_file = "lg_rpm_{:.2f}_img2.jpg"
    weights_path = "models/masking/lg_rpm_weights/lg_rpm_{}_{:.2f}.npy"
    res_fullpath = os.path.join(experiment_result, res_file)
    img1_fullpath = os.path.join(experiment_result, img1_file)
    img2_fullpath = os.path.join(experiment_result, img2_file)

    lg_rpm_ssim = []
    lg_rpm_mse = []
    lg_rpm_psnr = []
    lg_rpm_mask_time = []
    lg_rpm_demask_time = []
    lg_rpm_black_rate = []

    for p in all_p:
        resp_fullpath = res_fullpath.format(p)
        imgone_fullpath = img1_fullpath.format(p)
        imgtwo_fullpath = img2_fullpath.format(p)
        weight_fullpath = weights_path.format(length, p)

        # Эксперименты

        if not os.path.isfile(resp_fullpath):
            print("=== Испытание LgRPM: p = {} ===".format(p))
            model = LG_RPM(length=length, p=p, bandwidth=5)
            model.load(weight_fullpath)
            masker = MaskMaster(length=length, p=p)
            as_ = WorkerASDummy()
            kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                             ckpt_path="dependence/ckpt/kl-f16.ckpt")
            quant = WorkerQuantLinear()
            quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
            compress = WorkerCompressorDummy()
            codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

            img_flag = True

            with open(resp_fullpath, mode='w', newline='') as wf:
                csv_wr = csv.writer(wf)
                csv_wr.writerow(["id", "ssim", "mse", "psnr", "is_black", "mask_time", "demask_time"])
                id_ = 0

                for root, dirs, files in os.walk(dataset):
                    for file in files:
                        from_filepath = os.path.join(root, file)
                        image = cv2.imread(from_filepath)
                        latent_bytes = codec.encode_frame(image)

                        pre_latent = torch.frombuffer(latent_bytes, dtype=torch.uint8)
                        # pre_latent = pre_latent.to(device=device)
                        pre_latent = pre_latent.cpu().detach().numpy()
                        latent = pre_latent.reshape(1, -1)
                        # latent = masker.prepare_latent(pre_latent, typer=typer)
                        a = time.time()
                        mini_latent = masker.get_latent(latent)
                        b = time.time()
                        mask = model.eval(mini_latent)
                        numpy_img = masker.recompose(mini_latent, mask, batch_size=1)
                        # restore_latent = torch.tensor(restore_latent, device=device)
                        # restore_latent = masker.restore_latent(restore_latent)
                        c = time.time()

                        # restore_latent = restore_latent.to('cpu')
                        # numpy_img = restore_latent.numpy()
                        byter = numpy_img.tobytes()
                        res_image = codec.decode_frame(byter)

                        mask_time = round((b - a) * 1000, 2)  # мс
                        demask_time = round((c - b) * 1000, 2)  # мс

                        if img_flag:
                            cv2.imwrite(imgone_fullpath, image)
                            cv2.imwrite(imgtwo_fullpath, res_image)
                            img_flag = False

                        ssim = ssim_metric(image, res_image)
                        mse = mse_metric(image, res_image)
                        psnr = psnr_metric(image, res_image)
                        ssim = round(ssim, 2)
                        mse = round(mse, 2)
                        psnr = round(psnr, 2)

                        if res_image.any():
                            is_black = 0
                        else:
                            is_black = 1

                        csv_wr.writerow([id_, ssim, mse, psnr, is_black, mask_time, demask_time])
                        id_ += 1

        # Обработки

        lg_rpm_data = pandas.read_csv(resp_fullpath)
        ssim_data = lg_rpm_data["ssim"].to_numpy()
        mse_data = lg_rpm_data["mse"].to_numpy()
        psnr_data = lg_rpm_data["psnr"].to_numpy()
        masktime_data = lg_rpm_data["mask_time"].to_numpy()
        demasktime_data = lg_rpm_data["demask_time"].to_numpy()
        blackrate_data = lg_rpm_data["is_black"].to_numpy()

        ssim = round(ssim_data.mean(), 2)
        mse = round(mse_data.mean(), 2)
        psnr = round(psnr_data.mean(), 2)
        mask_time = round(masktime_data.mean(), 2)
        demask_time = round(demasktime_data.mean(), 2)
        black_rate = round(len(blackrate_data[blackrate_data == 1])/len(blackrate_data) * 100, 2)

        lg_rpm_ssim.append(ssim)
        lg_rpm_mse.append(mse)
        lg_rpm_psnr.append(psnr)
        lg_rpm_mask_time.append(mask_time)
        lg_rpm_demask_time.append(demask_time)
        lg_rpm_black_rate.append(black_rate)

if test_ln_rpm:
    res_file = "ln_rpm_{:.2f}.csv"
    img1_file = "ln_rpm_{:.2f}_img1.jpg"
    img2_file = "ln_rpm_{:.2f}_img2.jpg"
    weights_path = "models/masking/ln_rpm_weights/ln_rpm_{}_{:.2f}.npy"
    res_fullpath = os.path.join(experiment_result, res_file)
    img1_fullpath = os.path.join(experiment_result, img1_file)
    img2_fullpath = os.path.join(experiment_result, img2_file)

    ln_rpm_ssim = []
    ln_rpm_mse = []
    ln_rpm_psnr = []
    ln_rpm_mask_time = []
    ln_rpm_demask_time = []
    ln_rpm_black_rate = []

    for p in all_p:
        resp_fullpath = res_fullpath.format(p)
        imgone_fullpath = img1_fullpath.format(p)
        imgtwo_fullpath = img2_fullpath.format(p)
        weight_fullpath = weights_path.format(length, p)

        # Эксперименты

        if not os.path.isfile(resp_fullpath):
            print("=== Испытание LnRPM: p = {} ===".format(p))
            model = LN_RPM(length=length, p=p, bandwidth=5)
            model.load(weight_fullpath)
            masker = MaskMaster(length=length, p=p)
            as_ = WorkerASDummy()
            kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                             ckpt_path="dependence/ckpt/kl-f16.ckpt")
            quant = WorkerQuantLinear()
            quant.adjust_params(autoencoder_worker="AutoencoderKL_F16")
            compress = WorkerCompressorDummy()
            codec = NeuroCodec(as_=as_, vae=kl_f16, quant=quant, compressor=compress)

            img_flag = True

            with open(resp_fullpath, mode='w', newline='') as wf:
                csv_wr = csv.writer(wf)
                csv_wr.writerow(["id", "ssim", "mse", "psnr", "is_black", "mask_time", "demask_time"])
                id_ = 0

                for root, dirs, files in os.walk(dataset):
                    for file in files:
                        from_filepath = os.path.join(root, file)
                        image = cv2.imread(from_filepath)
                        latent_bytes = codec.encode_frame(image)

                        pre_latent = torch.frombuffer(latent_bytes, dtype=torch.uint8)
                        # pre_latent = pre_latent.to(device=device)
                        pre_latent = pre_latent.cpu().detach().numpy()
                        latent = pre_latent.reshape(1, -1)
                        # latent = masker.prepare_latent(pre_latent, typer=typer)
                        a = time.time()
                        mini_latent = masker.get_latent(latent)
                        b = time.time()
                        mask = model.eval(mini_latent)
                        numpy_img = masker.recompose(mini_latent, mask, batch_size=1)
                        # restore_latent = torch.tensor(restore_latent, device=device)
                        # restore_latent = masker.restore_latent(restore_latent)
                        c = time.time()

                        # restore_latent = restore_latent.to('cpu')
                        # numpy_img = restore_latent.numpy()
                        byter = numpy_img.tobytes()
                        res_image = codec.decode_frame(byter)

                        mask_time = round((b - a) * 1000, 2)  # мс
                        demask_time = round((c - b) * 1000, 2)  # мс

                        if img_flag:
                            cv2.imwrite(imgone_fullpath, image)
                            cv2.imwrite(imgtwo_fullpath, res_image)
                            img_flag = False

                        ssim = ssim_metric(image, res_image)
                        mse = mse_metric(image, res_image)
                        psnr = psnr_metric(image, res_image)
                        ssim = round(ssim, 2)
                        mse = round(mse, 2)
                        psnr = round(psnr, 2)

                        if res_image.any():
                            is_black = 0
                        else:
                            is_black = 1

                        csv_wr.writerow([id_, ssim, mse, psnr, is_black, mask_time, demask_time])
                        id_ += 1

        # Обработки

        ln_rpm_data = pandas.read_csv(resp_fullpath)
        ssim_data = ln_rpm_data["ssim"].to_numpy()
        mse_data = ln_rpm_data["mse"].to_numpy()
        psnr_data = ln_rpm_data["psnr"].to_numpy()
        masktime_data = ln_rpm_data["mask_time"].to_numpy()
        demasktime_data = ln_rpm_data["demask_time"].to_numpy()
        blackrate_data = ln_rpm_data["is_black"].to_numpy()

        ssim = round(ssim_data.mean(), 2)
        mse = round(mse_data.mean(), 2)
        psnr = round(psnr_data.mean(), 2)
        mask_time = round(masktime_data.mean(), 2)
        demask_time = round(demasktime_data.mean(), 2)
        black_rate = round(len(blackrate_data[blackrate_data == 1])/len(blackrate_data) * 100, 2)

        ln_rpm_ssim.append(ssim)
        ln_rpm_mse.append(mse)
        ln_rpm_psnr.append(psnr)
        ln_rpm_mask_time.append(mask_time)
        ln_rpm_demask_time.append(demask_time)
        ln_rpm_black_rate.append(black_rate)


if test_summary:
    plt.xlabel("p")
    plt.ylabel("SSIMср")
    plt.plot(all_p, bsrnn_ssim, label="НСВБП", color="red")
    plt.plot(all_p, sbsrnn_m_ssim, label="ПНСВБП-М", color="blue")
    plt.plot(all_p, sbsrnn_f_ssim, label="ПНСВБП-П", color="green")
    plt.plot(all_p, lg_rpm_ssim, label="ЛгРПМ", color="yellow")
    plt.plot(all_p, ln_rpm_ssim, label="ЛнРПМ", color="black")
    plt.legend(loc='upper right')
    # plt.axvline(x=0.10, color='r', linestyle='dashed')
    # plt.text(0.13, 0.05, "0,10", color='red')
    plt.savefig(r'D:\UserData\Работа\Проекты_статей\Маскирование\Маскирование2_рисунки\4.jpg', dpi=300)
    # plt.show()
    plt.close()

    plt.xlabel("p")
    plt.ylabel("PSNRср, дБ")
    plt.plot(all_p, bsrnn_psnr, label="НСВБП", color="red")
    plt.plot(all_p, sbsrnn_m_psnr, label="ПНСВБП-М", color="blue")
    plt.plot(all_p, sbsrnn_f_psnr, label="ПНСВБП-П", color="green")
    plt.plot(all_p, lg_rpm_psnr, label="ЛгРПМ", color="yellow")
    plt.plot(all_p, ln_rpm_psnr, label="ЛнРПМ", color="black")
    plt.legend(loc='upper right')
    # plt.axvline(x=0.10, color='r', linestyle='dashed')
    # plt.text(0.13, 0.05, "0,10", color='red')
    plt.savefig(r'D:\UserData\Работа\Проекты_статей\Маскирование\Маскирование2_рисунки\5.jpg', dpi=300)
    # plt.show()
    plt.close()

    plt.xlabel("p")
    plt.ylabel("tm, мс")
    plt.plot(all_p, bsrnn_mask_time, label="НСВБП", color="red")
    plt.plot(all_p, sbsrnn_m_mask_time, label="ПНСВБП-М", color="blue")
    plt.plot(all_p, sbsrnn_f_mask_time, label="ПНСВБП-П", color="green")
    plt.plot(all_p, lg_rpm_mask_time, label="ЛгРПМ", color="yellow")
    plt.plot(all_p, ln_rpm_mask_time, label="ЛнРПМ", color="black")
    plt.legend(loc='upper right')
    # plt.axvline(x=0.10, color='r', linestyle='dashed')
    # plt.text(0.13, 0.05, "0,10", color='red')
    plt.savefig(r'D:\UserData\Работа\Проекты_статей\Маскирование\Маскирование2_рисунки\6.jpg', dpi=300)
    # plt.show()
    plt.close()

    plt.xlabel("p")
    plt.ylabel("tdm, мс")
    plt.plot(all_p, bsrnn_demask_time, label="НСВБП", color="red")
    plt.plot(all_p, sbsrnn_m_demask_time, label="ПНСВБП-М", color="blue")
    plt.plot(all_p, sbsrnn_f_demask_time, label="ПНСВБП-П", color="green")
    plt.plot(all_p, lg_rpm_demask_time, label="ЛгРПМ", color="yellow")
    plt.plot(all_p, ln_rpm_demask_time, label="ЛнРПМ", color="black")
    plt.legend(loc='upper right')
    # plt.axvline(x=0.10, color='r', linestyle='dashed')
    # plt.text(0.13, 0.05, "0,10", color='red')
    plt.savefig(r'D:\UserData\Работа\Проекты_статей\Маскирование\Маскирование2_рисунки\7.jpg', dpi=300)
    # plt.show()
    plt.close()

    print("=== BLACK RATE ===")
    print("НСВБП:", bsrnn_black_rate)
    print("ПНСВБП-М:", sbsrnn_m_black_rate)
    print("ПНСВБП-П:", sbsrnn_f_black_rate)
    print("НСВБП:", lg_rpm_black_rate)
    print("НСВБП:", ln_rpm_black_rate)

    print()
    print("=== НСВБП ===")
    print("SSIM:", bsrnn_ssim)
    print("MSE:", bsrnn_mse)
    print("PSNR:", bsrnn_psnr)
    print("tm:", bsrnn_mask_time)
    print("tdm:", bsrnn_demask_time)


