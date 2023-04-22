import os
import time

from compress import run_coder, run_decoder
from utils import load_image, save_img, get_rescaled_cv2, \
    metrics_img, write_metrics_in_file, create_dir
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)

size = (512, 512)

dir_path = "data/input"
dir_name = "input"
output_rescaled_data_path = "rescaled"
test_path = "test"


def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False):
    start = time.time()  ## точка отсчета времени
    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}") 
    
    try:
        count = 0
        logger.debug(f"get files in dir = {dir_name}")
        
        # цикл обработки кадров
        for img_path, img_name in load_image(dir_path):

            # считывание кадра из input
            image = cv2.imread(img_path)
            count += 1
            logger.debug(f"compressing file {img_name} in dir {dir_name}; count = {count};" 
                            f" img size = {size} max 9")

            # создание директории для сохранения сжатого изображения и резултатов метрик
            if not os.path.exists(f"data/output/{count}_run"):
                create_dir(f"{count}_run")
            save_dir_name = f"{count}_run"

            # сжатие кадра для отправления на НС
            img = get_rescaled_cv2(image, size)
            if save_rescaled_out:
                save_img(img, path=save_dir_name, name_img=img_name)

            # функции НС
            run_coder()
            run_decoder()

            if is_save:
                save_img(img, path=save_dir_name, name_img=img_name)

            # сохранение метрик
            if save_metrics:   
                width = int(image.shape[1])
                height = int(image.shape[0])
                dim = (width, height)
                rescaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) 
                data = metrics_img(image, rescaled_img)
                write_metrics_in_file(f"data/output/{save_dir_name}", data, img_name)

    except Exception as err:
        logger.error(f"error is file {img_name} "
                        f"for write in dir {save_dir_name}\n error {err}")
    end = time.time() - start  ## собственно время работы программы
    logger.debug(f'Complete: {end}')

if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=False)
