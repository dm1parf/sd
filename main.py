import os
import warnings

import torch

from compress import ns_run
from utils import get_rescaled_img, get_rescaled_img_using_cv2, load_image, search_dir

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning


def create_dir (new_dir_name: str, index: str = ""):
    if not os.path.exists(f"data/output/{new_dir_name}{index}/"):
        os.makedirs(f"data/output/{new_dir_name}{index}/")


# size = [(360, 240), (720, 480), (960, 582), (1280, 720), (1920, 1080)]
size = [(512, 512)]


def default_main (is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False):
    """
       Сжатие изображений в директории и сохранение результатов с использованием
       нейронной сети NS в соответствии с аргументами функции.

       Args:
           is_quantize (bool): Определяет, должна ли быть выполнена квантизация в ходе компрессии.
           is_save (bool): Определяет, должны ли быть сохранены сжатые изображения.
           save_metrics (bool): по умолчанию равен True и определяет, должны ли сохранятся метрики сжатия.
           save_rescaled_out (bool): Определяет, должны ли сохраняться измененные по размеру изображения.

       Returns:
           None
   """
    for index in range(len(size)):
        print(f"compressing files for is_quantize = {str(is_quantize)}")
        for dir_path, dir_name in search_dir():
            try:
                count = 0
                print(f"get files in dir = {dir_name}")
                for img_path, img_name in load_image(dir_path):
                    if img_name.__contains__("DS_Store"):
                        continue

                    count += 1
                    save_dir_name = f"{dir_name}_{size[index][0]}_{size[index][1]}_{index}"
                    print(f"compressing file {img_name} in dir {dir_name}; count = {count};"
                          f" img size = {size[index]} max 9")

                    create_dir(f"{dir_name}_{size[index][0]}_{size[index][1]}", f"_{index}")
                    # img = get_rescaled_img(img_path, size[index])
                    img = get_rescaled_img_using_cv2(img_path, size[index])
                    ns_run(img=img, img_name=img_name.removesuffix('.jpg'),
                           dir_name=save_dir_name,
                           is_quantize=is_quantize, is_save=is_save, save_metrics=save_metrics,
                           save_rescaled_out=save_rescaled_out)

            except Exception as err:
                print(f"error is file {img_name} "
                      f"for write in dir {save_dir_name}\n error {err}")
                continue

                


if __name__ == '__main__':
    default_main(is_quantize=True, is_save=True, save_metrics=False, save_rescaled_out=True)
