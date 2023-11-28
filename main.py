import os
import time

import cv2

from common.logging_sd import configure_logger
from compress import run_coder, createSd
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, SHOW_VIDEO, is_save, is_quantize, \
    PREDICTION_MODEL_PATH, REAL, REAL_NAME, FAKE_NAME, FAKE, Platform, USE_OPTIMIZED_PREDICTION, \
    OPTIMIZED_PREDICTION_MODEL_PATH, DEVICE
from core import load_and_rescaled, latent_to_img
from prediction import Model, DMVFN
from prediction.model.models import DMVFN_optim
from utils import save_img, metrics_img, write_metrics_in_file

logger = configure_logger(__name__)
createSd(Platform.MAIN)


def default_main(save_metrics=True):
    start = time.time()  ## точка отсчета времени
    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT) and save_metrics:
        os.makedirs(DIR_PATH_OUTPUT)

    logger.debug(f"get files in dir = {DIR_NAME}")

    if SHOW_VIDEO:
        window_name = 'Video'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if USE_OPTIMIZED_PREDICTION:
        model = Model(
            DMVFN_optim(os.path.abspath(PREDICTION_MODEL_PATH),
                        os.path.abspath(OPTIMIZED_PREDICTION_MODEL_PATH),
                        DEVICE
                        ))
    else:
        model = Model(
            DMVFN(os.path.abspath(PREDICTION_MODEL_PATH), DEVICE))

    pattern = [REAL_NAME] * REAL + [FAKE_NAME] * FAKE

    restored_imgs = []

    for i, (rescaled_img, image, img_name, save_parent_dir_name, save_dir_name) in enumerate(load_and_rescaled(True)):

        # функции НС
        if pattern[i % len(pattern)] == REAL_NAME:

            coderTimeStart = time.time()
            compress_img = run_coder(cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2RGB))
            logger.debug(f"time spent on coder is {time.time() - coderTimeStart}")

            decoderTimeStart = time.time()
            uncompress_img = latent_to_img(compress_img)
            logger.debug(f"time spent on decoder is {time.time() - decoderTimeStart}")

        elif pattern[i % len(pattern)] == FAKE_NAME:
            predictTimeStart = time.time()
            uncompress_img = model.predict(restored_imgs[-2:])
            logger.debug(f"time spent on prediction is {time.time() - predictTimeStart}")

        restored_imgs.append(uncompress_img)

        if len(restored_imgs) > 2:
            del restored_imgs[0]

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
    if SHOW_VIDEO:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    default_main(save_metrics=True)
