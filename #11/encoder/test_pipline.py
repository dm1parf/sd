import os
import time


import os
import time

import cv2
import numpy as np
from PIL import Image
from numpy import ndarray
import logging


SCALED_SIZE_DEFAULT = (1200, 1200)
INPUT_DATA_PATH = "data/test"
OUTPUT_DATA_PATH = "../data/train/output"

SIZE = (512, 512)

DIR_PATH_INPUT = "../data/train/input"
DIR_PATH_OUTPUT = "../data/train/output"
DIR_NAME = "input"
TEST_PATH = "test"


def run_coder():
    ...


def run_decoder():
    ...




def create_dir(new_dir_name: str, index: str = ""):
        os.makedirs(f"{OUTPUT_DATA_PATH}/{new_dir_name}/")


def load_image(path: str = INPUT_DATA_PATH): 
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                f = os.path.join(path, filename)
                if os.path.isfile(f):
                    yield f, filename


def save_img(img, path: str, name_img: str = 'default'):
    if os.path.exists(f'{OUTPUT_DATA_PATH}/{path}'):
            cv2.imwrite(f'{OUTPUT_DATA_PATH}/{path}/0_{name_img}', img)


def get_rescaled_cv2(image, scaled_size: tuple = (512, 512)):
        res = cv2.resize(image, scaled_size)
        return res


def write_metrics_in_file(path: str, data: tuple, image_name: str, time: time):
        data_str = f"image_name: {image_name}\n" \
                   f"ssim_data = {data[0]}\n" \
                   f"pirson_data = {data[1]}\n" \
                   f"cosine_similarity = {data[2]}\n" \
                   f"mse = {data[3]}\n" \
                   f"hamming_distance = {data[4]}\n"\
                   f"frame_compression_time = {time}\n"
        with open(f"{path}/metrics.txt", mode='w') as f:
            f.write(data_str)


def metrics_img(image, denoised_img) -> tuple:
        img1 = (np.array(image).ravel())
        img2 = (np.array(denoised_img).ravel())
        ssim_data = metrics.ssim(image, denoised_img)
        pirson_data = metrics.cor_pirson(img1, img2)
        cosine_similarity = metrics.cosine_similarity_metric(img1, img2)
        mse = metrics.mse_metric(image, denoised_img)
        hamming_distance = metrics.hamming_distance_metric(image, denoised_img)
        result_metrics: tuple = (ssim_data, pirson_data, cosine_similarity, mse, hamming_distance)
        return result_metrics

def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False):
    start = time.time()  ## точка отсчета времени

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT):
        os.makedirs(DIR_PATH_OUTPUT)

    count = 0

    # цикл обработки кадров
    for img_path, img_name in load_image(DIR_PATH_INPUT):
        start = time.time()

        # считывание кадра из input
        image = cv2.imread(img_path)
        count += 1

        # создание директории для сохранения сжатого изображения и резултатов метрик
        if not os.path.exists(f"data/output/{count}_run"):
            create_dir(f"{count}_run")
        save_dir_name = f"{count}_run"

        # сжатие кадра для отправления на НС
        img = get_rescaled_cv2(image, SIZE)
        if save_rescaled_out:
            save_img(img, path=save_dir_name, name_img=img_name)

        # функции НС
        run_coder()
        run_decoder()

        end_time = time.time() - start

        if is_save:
            save_img(img, path=save_dir_name, name_img=img_name)


    end = time.time() - start  ## собственно время работы программы

if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=False)
