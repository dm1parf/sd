import os

import cv2
import numpy as np
from metrics import ssim, pirson
from PIL import Image
from numpy import ndarray
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] > %(message)s")
handler = logging.FileHandler(f"{__name__}.log", mode='w')
handler.setFormatter(formatter)
logger.addHandler(handler)


scaled_size_default = (1200, 1200)
input_data_path = "data/input"
output_data_path = "data/output"


def search_dir(path: str = input_data_path):
    for dir_name in os.listdir(path):
        f = os.path.join(path, dir_name)
        if os.path.isdir(f):
            yield f, dir_name


def load_image(path: str = input_data_path):
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            yield f, filename


def save_img(img, path: str, name_img: str = 'default'):
    if os.path.exists(f'{output_data_path}/{path}'):
        try:
            cv2.imwrite(f'{output_data_path}/{path}/0_{name_img}', img)
            logger.info(f"The compressed image {name_img} was "
                         f"saved successfully in the directory {output_data_path}/{path}")
        except:
            logger.error(f"Failed to save images {name_img} to the directory {output_data_path}/{path}, "
                          f"file is corrupted")
    else:
        logger.error(f"Failed to save images {name_img} to the directory {output_data_path}/{path}, catalog not found")


def rescaled_and_save(img, path: str, name_img='default', scaled_size: tuple = (1980, 1080)):
    if os.path.exists(f'{output_data_path}/{path}'):
        try:
            rescaled_img = cv2.resize(np.array(img), (1920, 1080))
            cv2.imwrite(f'{output_data_path}/{path}/0_{name_img}', rescaled_img)
        except:
            logger.error(f"Failed to save rescaled_images {name_img} to the directory {output_data_path}/{path}, "
                          f"file is corrupted")
    else:
        logger.error(f"Failed to save rescaled_images {name_img} to the directory {output_data_path}/{path},"
                      f"catalog not found")


def get_rescaled_cv2(img_path: str, scaled_size: tuple = (512, 512)):
    if os.path.exists(img_path):
        if img_path.__contains__('.png') or img_path.__contains__('.jpg'):
                image = cv2.imread(img_path)
                res = cv2.resize(image, scaled_size)
                logger.info(f'file compression in the "{img_path}" directory successfully')
                return res
    else:
        logger.error(f'file compression in the "{img_path}" directory failed')


def write_metrics_in_file(path: str, data: tuple, image_name: str):
    if os.path.exists(path):
        data_str = f"image_name: {image_name}\n" \
                   f"ssim_data = {data[0]}\n" \
                   f"pirson_data = {data[1]}\n"
        with open(path, mode='w') as f:
            f.write(data_str)
    else:
        logger.error(f"Failed to save metrics to the directory {path}, catalog not found")


def metrics_img(img_path, denoised_img) -> tuple:
    try:
        if img_path.__contains__('.png') or img_path.__contains__('.jpg'):
            image = cv2.imread(img_path)
            img1 = (np.array(image))

            width = int(image.shape[1])
            height = int(image.shape[0])
            dim = (width, height)

            # resize image
            rescaled_img = cv2.resize(denoised_img, dim, interpolation=cv2.INTER_AREA)
            img2 = (np.array(rescaled_img))
            ssim_data = 1 - ssim.ssim(img1, img2)
            pirson_data = pirson.cor_pirson(img1, img2)
            logger.info(f'Collecting image {img_path} metrics successfully')
            result_metrics: tuple = (ssim_data, pirson_data)
            return result_metrics
    except:
        logger.error(f'Failed to collect image {img_path} metrics', exc_info=True)


# it is testing run for util function
if __name__ == '__main__':
    for i, ii in search_dir():
        for j, jj in load_image(i):
            print(j)
