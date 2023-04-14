import os
import logging
import warnings

from utils import load_image, search_dir, save_img, rescaled_and_save, get_rescaled_cv2, \
    metrics_img, write_metrics_in_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] > %(message)s")
handler = logging.FileHandler(f"{__name__}.log", mode='w')
handler.setFormatter(formatter)
logger.addHandler(handler)

size = (512, 512)

output_rescaled_data_path = "rescaled"
output_metrics_data_path = "data/output/metrics_result/metrics_result.txt"
test_path = "test"


def default_main (is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False):

    logger.info(f"compressing files for is_quantize = {str(is_quantize)}")
    for dir_path, dir_name in search_dir():
        try:
            count = 0
            logger.info(f"get files in dir = {dir_name}")
            for img_path, img_name in load_image(dir_path):
                if img_name.__contains__("DS_Store"):
                    continue

                count += 1
                save_dir_name = f"{dir_name}"
                logger.info(f"compressing file {img_name} in dir {dir_name}; count = {count};" 
                             f" img size = {size} max 9")
                # img = get_rescaled_img(img_path, size[index])
                img = get_rescaled_cv2(img_path, size)
                if is_save:
                    save_img(img, path=save_dir_name, name_img=img_name)
                #
                # if save_rescaled_out:
                #     rescaled_and_save(img, path=output_rescaled_data_path, name_img=img_name)

                data = metrics_img(img_path, img)
                if save_metrics:
                    write_metrics_in_file(output_metrics_data_path, data, img_name)

        except Exception as err:
            logger.info(f"error is file {img_name} "
                  f"for write in dir {save_dir_name}\n error {err}")


if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=True)
