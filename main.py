import os
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from bonch_utils import load_image, metrics_img, save_img, get_rescaled_cv2, create_dir
from common.dir_utils import is_dir_empty
from common.logging_sd import configure_logger
from compress import run_coder, run_decoder
from constants.constant import DATA_PATH, INPUT_PATH, OUTPUT_PATH, SIZE


logger = configure_logger('main')


def create_exp_dir(dir, is_continue=None):
    if is_continue is not None:
        new_exp_dir = os.path.join(dir, f'exp_{is_continue}')
        return new_exp_dir

    dirs = os.listdir(dir)
    used_numbers = [int(dirname[len('exp_'):]) for dirname in dirs if 'exp_' in dirname]
    max_number = max(used_numbers) if len(used_numbers) > 0 else -1
    new_exp_dir = os.path.join(dir, f'exp_{max_number + 1}')
    os.mkdir(new_exp_dir)
    return new_exp_dir

def clone_input_dirs_stucture(input_path, output_path, create_results_folders=False):
    for dirpath, dirnames, filenames in os.walk(input_path):
        # logger.debug((dirpath, dirnames, filenames))
        # break
        # if dirnames
        # for dirname in dirnames:
        dirpath = dirpath[len(input_path):]
        dirpath = dirpath[1:] if dirpath.startswith('/') else dirpath
        new_dir_name = os.path.join(output_path, dirpath)
        os.makedirs(new_dir_name, exist_ok=True)
            # logger.debug(dirpath)
            # logger.debug(new_dir_name)

def get_next_frame(input_path: str) -> Tuple[int, str, str]:
    picture_formats_list = ['.jpg', '.png']
    videos_formats_list = ['.mov', '.avi', '.mkv', '.mp4']
    for dirpath, dirnames, filenames in os.walk(input_path):
        filenames : list[str]
        
        for filename in filenames:
            
            filename = os.path.join(dirpath, filename)
            for picture_format in picture_formats_list:
                if filename.endswith(picture_format):
                    logger.debug(f'pic: {filename}')
                    
                    frame = cv2.imread(filename)
                    index = 0
                    yield index, filename, dirpath, frame
            
            for video_format in videos_formats_list:
                if filename.endswith(video_format):
                    cap = cv2.VideoCapture(filename)
                    index = 0
                    while True:
                        ret, frame = cap.read()
                        index += 1
                        if not ret:
                            break
                        logger.debug(f'vid: {filename}')
                        yield index, filename, dirpath, frame


def default_main(is_quantize=True, is_save=False, save_metrics=True, save_rescaled_out=False, debug=False, do_save_video=True, is_continue=None):
    # is_continue = 32 # directory to continue
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

    if is_continue is not None:
        output_path = create_exp_dir(output_path, is_continue=is_continue)
        logger.debug(f'continuing {is_continue}')
    else:
        output_path = create_exp_dir(output_path)
    logger.debug(f'output: {output_path}')

    # write_exp_config_info(output_path, is_quantize) #TODO

    clone_input_dirs_stucture(input_path, output_path, create_results_folders=True)
    # return

    # for idx, frame_path, frame in get_next_frame(input_path):

    prev_frame_path = None

    for idx, frame_path, dirpath, frame in get_next_frame(input_path):
        result_path = os.path.join(output_path, frame_path[len(input_path) + 1:])
        result_img_name = f'{result_path}_{idx}_result.jpg'
        if os.path.exists(result_img_name):
            logger.debug(f'skipping {result_img_name} exists')
            continue
        preprocessed_frame = get_rescaled_cv2(frame, SIZE)
        compressed_frame = run_coder(cv2.cvtColor(preprocessed_frame, 
                                                    cv2.COLOR_BGR2RGB))
        uncompressed_frame = run_decoder(compressed_frame)

        postprocessed_frame = cv2.cvtColor(np.array(uncompressed_frame), cv2.COLOR_RGB2BGR)

        
        cv2.imwrite(result_img_name, postprocessed_frame)    
        logger.debug(result_img_name)

        # return
    # for dir_name in os.listdir(input_path):  # цикл обработки кадров
    #     if dir_name == '.DS_Store':
    #         continue
        
    #     current_input_folder = os.path.join(input_path, dir_name)
    #     logger.debug(f'Working in {current_input_folder}')
        
    #     current_input_subfolder = current_input_folder.split('/')[-1]
    #     current_output_folder = os.path.join(output_path, current_input_subfolder)
    #     current_frames_folder = os.path.join(current_output_folder, 'frames')
    #     os.makedirs(current_frames_folder)

    #     if do_save_video:
    #         fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #         video_result_path = os.path.join(current_output_folder, 'video.mp4')
    #         writer = cv2.VideoWriter(video_result_path, fourcc, 25, (512, 512))
        
    #     for idx, (image, img_name) in enumerate(load_image(f"{current_input_folder}"), start=1):
    #         logger.debug(f'Image: {idx}')

    #         # считывание кадра из input
            
    #         count += 1
    #         logger.debug(f"compressing file {img_name} in dir {current_input_folder}; count = {count};"
    #                      f" img size = {SIZE} max 9")

            
    #         # if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
    #         #     create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
    #         # os.makedirs()
    #         save_parent_dir_name = f"{dir_name}_run"

    #         # создание директории для сохранения сжатого изображения и резултатов метрик
    #         # if not os.path.exists(f"{current_output_folder}/{count}_run"):
    #         #     create_dir(f"{current_output_folder}", f"{count}_run")
    #         save_dir_name = f"{count}_run"
    #         img_save_folder = os.path.join(current_frames_folder, save_dir_name)
    #         os.makedirs(img_save_folder)
    #         # сжатие кадра для отправления на НС
    #         img = get_rescaled_cv2(image, SIZE)
    #         if save_rescaled_out:
    #             save_img(img, path=img_save_folder, name_img=f"resc_{img_name}")
    #         if do_save_video:    
    #             writer.write(img)
    #         # функции НС
    #         before_encode_time = time.time()
    #         compress_img = run_coder(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         encode_time = time.time() - before_encode_time
    #         # if is_save: TODO: it is code not work for old sd pipeline, but good work for sd inp pipeline
    #         #     print(compress_img[0].shape)
    #         #     print(compress_img[1].shape)
    #         #     print(compress_img[2].shape)
    #         #     image_1 = Image.fromarray(compress_img[0][0])
    #         #     image_1.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_1.png")
    #         #
    #         #     for i in range(compress_img[1].shape[0]):
    #         #         image_2 = Image.fromarray(compress_img[1][i])
    #         #         image_2.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_2_{i}.png")
    #         #     for i in range(compress_img[2].shape[0]):
    #         #         image_2 = Image.fromarray(compress_img[1][i])
    #         #         image_2.save(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/image_3_{i}.png")

    #         before_decode_time = time.time()
    #         uncompress_img = run_decoder(compress_img)
    #         decode_time = time.time() - before_decode_time

    #         logger.debug(uncompress_img)
    #         uncompress_img = cv2.cvtColor(np.array(uncompress_img), cv2.COLOR_RGB2BGR)

    #         if is_save:
    #             save_img(uncompress_img, path=img_save_folder, name_img=img_name)

    #         # сохранение метрик
    #         if save_metrics:
    #             width = int(image.shape[1])
    #             height = int(image.shape[0])
    #             dim = (width, height)
    #             rescaled_img = cv2.resize(uncompress_img, dim, interpolation=cv2.INTER_AREA)
    #             # data = metrics_img(image, rescaled_img, img_path,
    #             #                    f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}/0_{img_name}")
    #             # write_metrics_in_file(f"{DIR_PATH_OUTPUT}/{save_parent_dir_name}/{save_dir_name}", data, img_name,
    #             #                       end_time)

    #         logger.debug(f'Encode time: {encode_time}')
    #         logger.debug(f'Decode time: {decode_time}')


if __name__ == '__main__':
    logger.debug(f"SD compression main starting...")

    default_main(is_quantize=True, is_save=True, save_metrics=True, save_rescaled_out=True)
