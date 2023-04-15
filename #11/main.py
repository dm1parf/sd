import os
import logging
import warnings
import time

from compress import run_ns
from utils import load_image, search_dir, save_img, rescaled_and_save, get_rescaled_cv2, \
    metrics_img, write_metrics_in_file
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)

size = (512, 512)

output_rescaled_data_path = "rescaled"
output_metrics_data_path = "data/output/metrics_result/metrics_result.txt"
test_path = "test"


def create_dir(new_dir_name: str, index: str = ""):
    if not os.path.exists(f"data/output/{new_dir_name}/"):
        os.makedirs(f"data/output/{new_dir_name}/")



def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False):
    start = time.time()  ## точка отсчета времени
    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

    for dir_path, dir_name in search_dir():
        try:
            count = 0
            logger.debug(f"get files in dir = {dir_name}")
            for img_path, img_name in load_image(dir_path):
                if img_name.__contains__("DS_Store"):
                    continue

                count += 1
                save_dir_name = f"{dir_name}"
                logger.debug(f"compressing file {img_name} in dir {dir_name}; count = {count};" 
                             f" img size = {size} max 9")
                create_dir(f"{dir_name}")

                if os.path.exists(img_path):
                    if img_path.__contains__('.png') or img_path.__contains__('.jpg'):
                        image = cv2.imread(img_path)

                img = get_rescaled_cv2(image, size)
                if save_rescaled_out:
                    save_img(img, path=save_dir_name, name_img=img_name)

                run_ns()

                width = int(image.shape[1])
                height = int(image.shape[0])
                dim = (width, height)
                rescaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                data = metrics_img(image, rescaled_img)
                if save_metrics:
                    write_metrics_in_file(output_metrics_data_path, data, img_name)
        except Exception as err:
            logger.error(f"error is file {img_name} "
                         f"for write in dir {save_dir_name}\n error {err}")
    end = time.time() - start  ## собственно время работы программы
    logger.info(f'Complete: {end}')

if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=False)
