import os

import cv2
import numpy as np
from metrics import metrics
from common.logging_sd import configure_logger
from PIL import Image
from numpy import ndarray
import logging


logger = configure_logger(__name__)


scaled_size_default = (1200, 1200)
input_data_path = "data/input"
output_data_path = "data/output"


def create_data_dir():
    os.makedirs(f"data/output/")



def create_dir(new_dir_name: str, index: str = ""):
    os.makedirs(f"data/output/{new_dir_name}/")

    

def search_dir(path: str = input_data_path):
    for dir_name in os.listdir(path):
        f = os.path.join(path, dir_name)
        if os.path.isdir(f):
            yield f, dir_name
        else:
            yield path, "input"
        


def load_image(path: str = input_data_path):
    for filename in os.listdir(path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            f = os.path.join(path, filename)
            logger.debug(f"{f}, {filename}")
            if os.path.isfile(f):
                yield f, filename


def save_img(img, path: str, name_img: str = 'default'):
    if os.path.exists(f'{output_data_path}/{path}'):
        try:
            cv2.imwrite(f'{output_data_path}/{path}/0_{name_img}', img)
            logger.debug(f"The compressed image {name_img} was "
                         f"saved successfully in the directory {output_data_path}/{path}")
        except:
            logger.error(f"Failed to save images {name_img} to the directory {output_data_path}/{path}, "
                          f"file is corrupted")
    else:
        logger.error(f"Failed to save images {name_img} to the directory {output_data_path}/{path}, catalog not found")


def get_rescaled_cv2(image, scaled_size: tuple = (512, 512)):
    try:
        res = cv2.resize(image, scaled_size)
        logger.debug(f'file compression in the directory successfully')
        return res
    except:
        logger.error(f'file compression in the directory failed')


def write_metrics_in_file(path: str, data: tuple, image_name: str):
    if os.path.exists(path):
        data_str = f"image_name: {image_name}\n" \
                   f"ssim_data = {data[0]}\n" \
                   f"pirson_data = {data[1]}\n" \
                   f"cosine_similarity = {data[2]}\n" \
                   f"mse = {data[3]}\n" \
                   f"hamming_distance = {data[4]}\n"
        with open(f"{path}/metrics.txt", mode='w') as f:
            f.write(data_str)
    else:
        logger.error(f"Failed to save metrics to the directory {path}, catalog not found")


def metrics_img(image, denoised_img) -> tuple:
    try:
        img1 = (np.array(image).ravel())
        img2 = (np.array(denoised_img).ravel())
        ssim_data = metrics.ssim(image, denoised_img)
        pirson_data = metrics.cor_pirson(img1, img2)
        cosine_similarity = metrics.cosine_similarity_metric(img1, img2)
        mse = metrics.mse_metric(image, denoised_img)
        hamming_distance = metrics.hamming_distance_metric(image, denoised_img)
        logger.debug(f'Collecting image metrics successfully')
        result_metrics: tuple = (ssim_data, pirson_data, cosine_similarity, mse, hamming_distance)
        return result_metrics
    except:
        logger.error(f'Failed to collect image metrics', exc_info=True)


# it is testing run for util function
if __name__ == '__main__':
    for i, ii in search_dir():
        for j, jj in load_image(i):
            print(j)
