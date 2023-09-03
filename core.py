import os
import time

import numpy as np

from compress import run_decoder
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, SIZE, USE_VIDEO, save_rescaled_out
from utils import load_image, save_img, get_rescaled_cv2, create_dir,load_frame_video
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)


def load_and_rescaled():
    count = 0
    load_frame = load_image if not USE_VIDEO else load_frame_video

    for dir_name in os.listdir(DIR_PATH_INPUT):  # цикл обработки кадров
        # for img_path, img_name in load_frame(f"{DIR_PATH_INPUT}/{dir_name}"):
        for params_frame in load_frame(f"{DIR_PATH_INPUT}/{dir_name}"):

            if not USE_VIDEO:
                img_path, img_name = params_frame
            else:
                frame, img_name, video_name = params_frame

            # считывание кадра из input
            if USE_VIDEO:
                image = frame
            else:
                image = cv2.imread(img_path)
            count += 1
            logger.debug(f"compressing file {img_name} in dir {DIR_NAME}; count = {count};"
                         f" img size = {SIZE} max 9")

            # сжатие кадра для отправления на НС
            img = get_rescaled_cv2(image, SIZE)

            if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
                create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
            save_parent_dir_name = f"{dir_name}_run"

            # создание директории для сохранения сжатого изображения и резултатов метрик
            if not os.path.exists(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{count}_run"):
                create_dir(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}", f"{count}_run")
            save_dir_name = f"{count}_run"

            if save_rescaled_out:
                save_img(img, path=f"{save_parent_dir_name}/{save_dir_name}", name_img=f"resc_{img_name}")

            yield img, image, img_name, save_parent_dir_name, save_dir_name


def latent_to_img(compress_img):
    uncompress_img = run_decoder(compress_img)
    uncompress_img = cv2.cvtColor(np.array(uncompress_img), cv2.COLOR_RGB2BGR)
    return uncompress_img

