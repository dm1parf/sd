import argparse
from models.restoration.ncirnn import NCIRNN
from utils.workers import WorkerSRDummy, WorkerASMoveDistribution
import torch
import copy
import numpy as np
import cv2
from skimage.metrics import structural_similarity
import math
import os
import time
import random


arguments = argparse.ArgumentParser(prog="Обучение NCIRNN",
                                    description="Восстановление изображений.")
arguments.add_argument("-c", dest="cfg", type=int, default=1, help="Конфигурация")
args = arguments.parse_args()
cfg = args.cfg

dataset_source = "dataset_preparation/spbsut_25fps_full"
dataset_cfg = "dataset_preparation/cfg{}_spbsut_25_full".format(cfg)
nominal_type = torch.float16
worker_restore = WorkerASMoveDistribution()
worker_sr = WorkerSRDummy()

# model = NCIRNN()
model = torch.jit.load("ncirnn_cfg1_1801.ts")
model.to(device="cuda", dtype=nominal_type)
model.eval()


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


good_ext = [".jpg", ".png", ".bmp"]
def find_random_file(pather):
    catalogue = os.listdir(pather)
    new_node = random.sample(catalogue, 1)[0]
    new_path = os.path.join(pather, new_node)
    if os.path.isfile(new_path):
        if os.path.splitext(new_path)[1] in good_ext:
            return new_path
        else:
            return None
    else:
        return find_random_file(new_path)


source_filepath = None
while source_filepath is None:
    source_filepath = find_random_file(dataset_source)

rel_path = os.path.relpath(source_filepath, dataset_source)
old_path = os.path.splitext(os.path.join(dataset_cfg, rel_path))[0] + ".jpg"


source_image = cv2.imread(source_filepath)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if source_image.shape[::-1][1:] != (1280, 720):
    image = cv2.resize(source_image, (1280, 720), interpolation=cv2.INTER_AREA)
cfg_image_ = cv2.imread(old_path)
cfg_image = np.copy(cfg_image_)
cfg_tensor, _ = worker_restore.prepare_work(cfg_image_, dest_type=nominal_type)


def decoding(model, tens):
    image_pixel = model(tens)
    image_restore, _ = worker_restore.restore_work(image_pixel)
    image_restore, _ = worker_sr.sr_work(image_restore, dest_size=[1280, 720])

    return image_restore


new_model_img = decoding(model, cfg_tensor)

ssim_true = ssim_metric(cfg_image, source_image)
mse_true = mse_metric(cfg_image, source_image)
psnr_true = psnr_metric(cfg_image, source_image)
ssim_test = ssim_metric(new_model_img, source_image)
mse_test = mse_metric(new_model_img, source_image)
psnr_test = psnr_metric(new_model_img, source_image)


print("--- До восстановления ---")
print("SSIM = {:.02f}".format(ssim_true))
print("MSE = {:.04f}".format(mse_true))
print("PSNR = {:.04f} дБ".format(psnr_true))
print()
print("--- После восстановления ---")
print("SSIM = {:.02f}".format(ssim_test))
print("MSE = {:.04f}".format(mse_test))
print("PSNR = {:.04f} дБ".format(psnr_test))


cv2.imshow("SOURCE FRAME", source_image)
cv2.imshow("WITHOUT RESTORATION", cfg_image)
cv2.imshow("WITH RESTORATION", new_model_img)
# dimg = cv2.vconcat([source_image, cfg_image, new_model_img])
# cv2.imshow("SOURCE - BEFORE - WITH RESTORATION", dimg)
cv2.waitKey(0)
