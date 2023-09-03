import os
import time
from PIL import Image
import cv2
import numpy as np
import traceback

from constants.constant import DIR_PATH_OUTPUT, INPUT_DATA_PATH_FROM_UTILS, USE_VIDEO
from metrics import metrics
from common.logging_sd import configure_logger

logger = configure_logger(__name__)


def create_dir(target_path: str, new_dir_name: str, index: str = ""):
    try:
        os.makedirs(f"{target_path}/{new_dir_name}/")
    except FileNotFoundError:
        logger.error(f"failed to create directory {DIR_PATH_OUTPUT}/{new_dir_name}", exc_info=True)


def load_frame_video(path: str = INPUT_DATA_PATH_FROM_UTILS):
    try:
        for filename in os.listdir(path):
            if filename.endswith('.mp4'):
                f = os.path.join(path, filename)
                if os.path.isfile(f):
                    cap = cv2.VideoCapture(f)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    try:
                        for i in range(frame_count):
                            ret, frame = cap.read()
                            if ret:
                                if i == range(frame_count):
                                    cap.release()
                                img_name = "{:04d}".format(i) + '.jpg'
                                yield frame, img_name, filename
                            else:
                                cap.release()
                    except Exception:
                        print(traceback.format_exc())

        logger.debug(f"Frame search success in {path} directory")
    except FileNotFoundError:
        logger.error(f"Error while reading image from directory {path}, catalog not found", exc_info=True)


def load_image(path: str = INPUT_DATA_PATH_FROM_UTILS):
    try:
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                f = os.path.join(path, filename)
                logger.debug(f"{f}, {filename}")
                if os.path.isfile(f):
                    yield f, filename
        logger.debug(f"Frame search success in {path} directory")
    except FileNotFoundError:
        logger.error(f"Error while reading image from directory {path}, catalog not found", exc_info=True)


def save_img(img, path: str, name_img: str = 'default'):
    if os.path.exists(f'{DIR_PATH_OUTPUT}/{path}'):
        try:
            cv2.imwrite(f'{DIR_PATH_OUTPUT}/{path}/0_{name_img}', img)
            logger.debug(f"The compressed image {name_img} was "
                         f"saved successfully in the directory {DIR_PATH_OUTPUT}/{path}")
        except FileNotFoundError:
            logger.error(f"Failed to save images {name_img} to the directory {DIR_PATH_OUTPUT}/{path}, "
                         f"file is corrupted", exc_info=True)
    else:
        logger.error(f"Failed to save images {name_img} to the directory {DIR_PATH_OUTPUT}/{path}, catalog not found",
                     exc_info=True)


def get_rescaled_cv2(image, scaled_size: tuple = (512, 512)):
    res = cv2.resize(image, scaled_size)
    logger.debug(f'file compression in the directory successfully')
    return res

def write_metrics_in_file(path: str, data: tuple, image_name: str, time: time):
    try:
        data_str = f"image_name: {image_name}\n" \
                   f"ssim_data = {data[0]}\n" \
                   f"pirson_data = {data[1]}\n" \
                   f"cosine_similarity = {data[2]}\n" \
                   f"mse = {data[3]}\n" \
                   f"hamming_distance = {data[4]}\n" \
                   f"lpips = {data[5]}\n" \
                   f"erqa = {data[6]}\n" \
                   f"y_msssim = {data[7].real}\n" \
                   f"y_psnr = {data[8]}\n" \
                   f"y_ssim = {data[9]}\n" \
                   f"lossless_compression = {os.path.getsize(f'{path}/compress_{image_name[:-4]}')}\n" \
                   f"frame_compression_time = {time}\n"
        with open(f"{path}/metrics.txt", mode='w') as f:
            f.write(data_str)
    except FileNotFoundError:
        logger.error(f"Failed to save metrics to the directory {path}, catalog not found", exc_info=True)


def metrics_img(image, denoised_img) -> tuple:

    img1 = (np.array(image).ravel())
    img2 = (np.array(denoised_img).ravel())
    to_pil1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    to_pil2 = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
    pil1 = Image.fromarray(to_pil1)
    pil2 = Image.fromarray(to_pil2)

    ssim_data = metrics.ssim(image, denoised_img)
    pirson_data = metrics.cor_pirson(img1, img2)
    cosine_similarity = metrics.cosine_similarity_metric(img1, img2)
    mse = metrics.mse_metric(image, denoised_img)
    hamming_distance = metrics.hamming_distance_metric(image, denoised_img)
    lpips = metrics.lpips_metric(pil1, pil2)
    # vmaf = metrics.vmaf(path_img, path_denoised_img)
    erqa = metrics.erqa_metrics(image, denoised_img)
    y_msssim = metrics.msssim(image, denoised_img)
    y_psnr = metrics.yuv_psnr_metric(image, denoised_img)
    y_ssim = metrics.yuv_ssim_metric(image, denoised_img)

    logger.debug(f'Collecting image metrics successfully')
    result_metrics: tuple = (ssim_data, pirson_data, cosine_similarity, mse, hamming_distance, lpips,
                             erqa, y_msssim, y_psnr, y_ssim)
    return result_metrics
