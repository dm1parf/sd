import argparse
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import math
import csv


encoder_dir = "source_frames"
decoder_dir = "dest_frames"
output = "quality_neuro.csv"

parser = argparse.ArgumentParser(prog="Измеритель качества кадров нейрокодека", description="Измеряет качество восстановления кадров")
parser.add_argument('-s', '--source', dest="source", type=str, default=encoder_dir)
parser.add_argument('-d', '--dest', dest="dest", type=str, default=decoder_dir)
parser.add_argument('-o', '--output', dest="output", type=str, default=output)
args = parser.parse_args()

encoder_dir = args.source
decoder_dir = args.dest
output = args.output

supported_images = [".jpg", ".png", ".bmp"]


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


def get_valid_list(test_dir):
    test_list = os.listdir(test_dir)
    test_list = [i for i in test_list if any([i.lower().endswith(j) for j in supported_images])]

    return test_list


def set_names(test_list):
    new_list = [os.path.splitext(i)[0] for i in test_list]
    new_set = set(new_list)
    return new_set


def filter_common_names(test_list, common_set):
    new_list = [i for i in test_list if os.path.splitext(i)[0] in common_set]
    return new_list


source_list = get_valid_list(encoder_dir)
dest_list = get_valid_list(decoder_dir)

source_set = set_names(source_list)
dest_set = set_names(dest_list)
common_set = source_set & dest_set

source_list = filter_common_names(source_list, common_set)
dest_list = filter_common_names(dest_list, common_set)

final_list = sorted(list(set(source_list) & set(dest_list)))

all_ssim = []
all_mse = []
all_psnr = []
with open(output, mode='w', encoding='utf-8', newline='') as wf:
    wcsf = csv.writer(wf)
    wcsf.writerow(["name", "ssim", "mse", "psnr"])

    for filer in final_list:
        source_name = os.path.join(encoder_dir, filer)
        dest_name = os.path.join(decoder_dir, filer)

        source_frame = cv2.imread(source_name)
        dest_frame = cv2.imread(dest_name)

        if dest_frame.shape != source_frame.shape:
            source_height, source_width, _ = source_frame.shape
            print("Несоответствие размерностей:", dest_frame.shape, "и", source_frame.shape, ".")
            print("Используем бикубическую интерполяцию...")
            dest_frame = cv2.resize(dest_frame, [source_width, source_height], interpolation=cv2.INTER_CUBIC)

        ssim = ssim_metric(source_frame, dest_frame)
        mse = mse_metric(source_frame, dest_frame)
        psnr = psnr_metric(source_frame, dest_frame)

        all_ssim.append(ssim)
        all_mse.append(mse)
        all_psnr.append(psnr)

        wcsf.writerow([filer, ssim, mse, psnr])

all_ssim = np.array(all_ssim)
all_mse = np.array(all_mse)
all_psnr = np.array(all_psnr)

ssim_mean = all_ssim.mean()
mse_mean = all_mse.mean()
psnr_mean = all_psnr.mean()

print("==========")
print("SSIMср = {}".format(ssim_mean))
print("MSEср = {}".format(mse_mean))
print("PSNRср = {}".format(psnr_mean))
print("==========")


