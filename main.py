import os
import time

from compress import run_coder, run_decoder
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, SIZE
from utils import load_image, save_img, get_rescaled_cv2, metrics_img, write_metrics_in_file, create_dir
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

    # цикл обработки кадров
    for img_path, img_name in load_image(DIR_PATH_INPUT):
        start = time.time()

        # считывание кадра из input
        image = cv2.imread(img_path)
        count += 1
        logger.debug(f"compressing file {img_name} in dir {DIR_NAME}; count = {count};"
                     f" img size = {SIZE} max 9")

        # создание директории для сохранения сжатого изображения и резултатов метрик
        if not os.path.exists(f"data/output/test_1_frames2/{count}_run"):
            create_dir(f"{count}_run")
        save_dir_name = f"{count}_run"

        # сжатие кадра для отправления на НС
        img = get_rescaled_cv2(image, SIZE)
        if save_rescaled_out:
            save_img(img, path=save_dir_name, name_img=img_name)

        # функции НС
        compress_img = run_coder(img)
        uncompress_img = run_decoder(compress_img)

        end_time = time.time() - start

        if is_save:
            save_img(uncompress_img, path=save_dir_name, name_img=img_name)

        # сохранение метрик
        if save_metrics:
            width = int(image.shape[1])
            height = int(image.shape[0])
            dim = (width, height)
            rescaled_img = cv2.resize(uncompress_img, dim, interpolation=cv2.INTER_AREA)
            data = metrics_img(image, rescaled_img)
            write_metrics_in_file(f"data/output/test_1_frames2/{save_dir_name}", data, img_name, end_time)

    end = time.time() - start  ## собственно время работы программы
    logger.debug(f'Complete: {end}')


if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=False)
