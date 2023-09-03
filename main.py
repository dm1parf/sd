import os
import time

import numpy as np

from compress import run_coder
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, SHOW_VIDEO, is_save, is_quantize, \
    save_rescaled_out, is_save_compress_bin
from utils import save_img, metrics_img, write_metrics_in_file
from core import load_and_rescaled, latent_to_img
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)


def default_main(save_metrics=True):
    start = time.time()  ## точка отсчета времени
    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT):
        os.makedirs(DIR_PATH_OUTPUT)

    logger.debug(f"get files in dir = {DIR_NAME}")

    if SHOW_VIDEO:
        window_name = 'Video'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for rescaled_img, image, img_name, save_parent_dir_name, save_dir_name in load_and_rescaled():
        # функции НС
        compress_img = run_coder(cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2RGB))

        if is_save_compress_bin:
            with open(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/compress_{img_name[:-4]}", 'wb') as f:
                f.write(compress_img)

        uncompress_img = latent_to_img(compress_img)

        if is_save:
            save_img(uncompress_img, path=f"{save_parent_dir_name}/{save_dir_name}", name_img=img_name)

        if SHOW_VIDEO:
            cv2.imshow(window_name, uncompress_img)
            cv2.waitKey(25)

        end_time = time.time() - start

        # сохранение метрик
        if save_metrics:
            width = int(image.shape[1])
            height = int(image.shape[0])
            dim = (width, height)
            rescaled_img = cv2.resize(uncompress_img, dim, interpolation=cv2.INTER_AREA)
            data = metrics_img(image, rescaled_img)
            write_metrics_in_file(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}", data, img_name, end_time)

    end = time.time() - start  ## собственно время работы программы
    logger.debug(f'Complete: {end}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    default_main(save_metrics=True)
