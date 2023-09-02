import os
import time

import cv2
import numpy as np
from PIL import Image

from bonch_utils import load_image, save_img, get_rescaled_cv2, create_dir
from common.dir_utils import is_dir_empty
from common.logging_sd import configure_logger
from compress import run_coder, run_decoder
from constants.constant import DATA_PATH, INPUT_PATH, OUTPUT_PATH, SIZE


logger = configure_logger('main')


def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False, do_save_video=True):
    input_path = os.path.join(DATA_PATH, INPUT_PATH)
    output_path = os.path.join(DATA_PATH, OUTPUT_PATH)

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if is_dir_empty(input_path):
        logger.info(f'Input dir is empty! Exiting....')
        return
    
    
    start = time.time()  ## точка отсчета времени
    
    logger.info(f"Quantization is {'ON' if is_quantize else 'OFF'}")

    count = 0
    logger.info(f"Input directory provided: {input_path}")

    for dir_name in os.listdir(input_path):  # цикл обработки кадров
        if dir_name == '.DS_Store':
            continue
        
        current_input_folder = os.path.join(input_path, dir_name)
        logger.debug(f'Working in {current_input_folder}')
        
        current_input_subfolder = current_input_folder.split('/')[-1]
        current_output_folder = os.path.join(output_path, current_input_subfolder)
        current_frames_folder = os.path.join(current_output_folder, 'frames')
        os.makedirs(current_frames_folder)

        if do_save_video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_result_path = os.path.join(current_output_folder, 'video.mp4')
            writer = cv2.VideoWriter(video_result_path, fourcc, 25, (512, 512))
        
        for idx, (image, img_name) in enumerate(load_image(f"{current_input_folder}"), start=1):
            logger.debug(f'Image: {idx}')

            # считывание кадра из input
            
            count += 1
            logger.debug(f"compressing file {img_name} in dir {current_input_folder}; count = {count};"
                         f" img size = {SIZE} max 9")

            
            # if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
            #     create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
            # os.makedirs()
            save_parent_dir_name = f"{dir_name}_run"

            # создание директории для сохранения сжатого изображения и резултатов метрик
            # if not os.path.exists(f"{current_output_folder}/{count}_run"):
            #     create_dir(f"{current_output_folder}", f"{count}_run")
            save_dir_name = f"{count}_run"
            img_save_folder = os.path.join(current_frames_folder, save_dir_name)
            os.makedirs(img_save_folder)
            # сжатие кадра для отправления на НС
            img = get_rescaled_cv2(image, SIZE)
            if save_rescaled_out:
                save_img(img, path=img_save_folder, name_img=f"resc_{img_name}")
            if do_save_video:    
                writer.write(img)
            # функции НС
            before_encode_time = time.time()
            compress_img = run_coder(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            encode_time = time.time() - before_encode_time
            # if is_save: TODO: it is code not work for old sd pipeline, but good work for sd inp pipeline
            #     print(compress_img[0].shape)
            #     print(compress_img[1].shape)
            #     print(compress_img[2].shape)
            #     image_1 = Image.fromarray(compress_img[0][0])
            #     image_1.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_1.png")
            #
            #     for i in range(compress_img[1].shape[0]):
            #         image_2 = Image.fromarray(compress_img[1][i])
            #         image_2.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_2_{i}.png")
            #     for i in range(compress_img[2].shape[0]):
            #         image_2 = Image.fromarray(compress_img[1][i])
            #         image_2.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_3_{i}.png")

            before_decode_time = time.time()
            uncompress_img = run_decoder(compress_img)
            decode_time = time.time() - before_decode_time

            logger.debug(uncompress_img)
            uncompress_img = cv2.cvtColor(np.array(uncompress_img), cv2.COLOR_RGB2BGR)

            if is_save:
                save_img(uncompress_img, path=img_save_folder, name_img=img_name)

            # сохранение метрик
            if save_metrics:
                width = int(image.shape[1])
                height = int(image.shape[0])
                dim = (width, height)
                rescaled_img = cv2.resize(uncompress_img, dim, interpolation=cv2.INTER_AREA)
                # data = metrics_img(image, rescaled_img, img_path,
                #                    f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/0_{img_name}")
                # write_metrics_in_file(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}", data, img_name,
                #                       end_time)

            logger.debug(f'Encode time: {encode_time}')
            logger.debug(f'Decode time: {decode_time}')


if __name__ == '__main__':
    logger.debug(f"SD compression main starting...")
    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=True)
