import os
import time

import numpy as np

from compress import run_coder, run_decoder
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, SIZE, USE_VIDEO, SHOW_VIDEO
from utils import load_image, save_img, get_rescaled_cv2, metrics_img, write_metrics_in_file, create_dir,load_frame_video
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)


def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False):
    start = time.time()  ## точка отсчета времени
    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT):
        os.makedirs(DIR_PATH_OUTPUT)

    count = 0
    logger.debug(f"get files in dir = {DIR_NAME}")

    if SHOW_VIDEO:
        window_name = 'Video'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    load_frame = load_image if not USE_VIDEO else load_frame_video

    for dir_name in os.listdir(DIR_PATH_INPUT):   # цикл обработки кадров
        # for img_path, img_name in load_frame(f"{DIR_PATH_INPUT}/{dir_name}"):
        for params_frame in load_frame(f"{DIR_PATH_INPUT}/{dir_name}"):
            start = time.time()

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

            if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
                create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
            save_parent_dir_name = f"{dir_name}_run"

            # создание директории для сохранения сжатого изображения и резултатов метрик
            if not os.path.exists(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{count}_run"):
                create_dir(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}", f"{count}_run")
            save_dir_name = f"{count}_run"

            # сжатие кадра для отправления на НС
            img = get_rescaled_cv2(image, SIZE)
            if save_rescaled_out:
                save_img(img, path=f"{save_parent_dir_name}/{save_dir_name}", name_img=f"resc_{img_name}")

            # функции НС
            compress_img = run_coder(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            uncompress_img = run_decoder(compress_img)
            uncompress_img = cv2.cvtColor(np.array(uncompress_img), cv2.COLOR_RGB2BGR)

            if SHOW_VIDEO:
                cv2.imshow(window_name, uncompress_img)
                cv2.waitKey(25)

            end_time = time.time() - start

            if is_save:
                save_img(uncompress_img, path=f"{save_parent_dir_name}/{save_dir_name}", name_img=img_name)

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
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=True)
