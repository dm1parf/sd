import os

import cv2
import numpy as np
from PIL import Image
from numpy import ndarray

scaled_size_default = (1200, 1200)
input_data_path = "data/input"
output_data_path = "data/output"


def search_dir (path: str = input_data_path):
    for dir_name in os.listdir(path):
        f = os.path.join(path, dir_name)
        if os.path.isdir(f):
            yield f, dir_name


def load_image (path: str = input_data_path):
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            yield f, filename


def get_rescaled_img (img_path: str, scaled_size=scaled_size_default):
    if img_path.__contains__('.png') or img_path.__contains__('.jpg'):
        img = Image.open(img_path)
        # img = ImageOps.fit(img, scaled_size, Image.ANTIALIAS)
        img.thumbnail(scaled_size, Image.ANTIALIAS)
        return img
    else:
        return None


def get_rescaled_img_using_cv2 (img_path: str, scaled_size=scaled_size_default):
    if img_path.__contains__('.png') or img_path.__contains__('.jpg'):
        image = cv2.imread(img_path)
        res = cv2.resize(image, scaled_size)
        return res


def rescaled_pil_img_and_write_using_cv2 (img, path):
    rescaled_img = cv2.resize(np.array(img), (1920, 1080))
    cv2.imwrite(path, rescaled_img)


def write_txt (path, data: ndarray):
    data.tofile(path, sep=" ", format="%s")


def write_metrics (path: str, data: str):
    with open(path, 'w') as f:
        f.write(data)


# it is testing run for util function
if __name__ == '__main__':
    for i, ii in search_dir():
        for j, jj in load_image(i):
            print(j)
