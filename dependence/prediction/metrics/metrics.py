import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
import cv2
import math
from common.utils import write_result
from common.logging import configure_logger
from scipy.spatial.distance import hamming

logger = configure_logger(__name__)


def cosine_similarity_metric(image1, image2):

    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
        [image1_vector], [image2_vector])[0][0]

    return similarity_score


def hamming_distance_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    hamming_distance = hamming(image1.flatten(), image2.flatten())

    return hamming_distance


def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse


def psnr_metric(image1, image2):
    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def ssim_metric(image1, image2):
    """Needs checking! Probably wrong!"""
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(
        image1, image2, data_range=(image2.max() - image2.min()))

    return score


def run_metrics(img1, img2, exp_dir, exp_no):
    out_file_path = f'{exp_dir}/metrics.txt'
    with open(out_file_path, 'a') as out_file:
        write_result(f'Run #{exp_no}', logger, out_file)

        mse_res = mse_metric(img1, img2)
        write_result(f'MSE: {mse_res}', logger, out_file)

        ssim_res = ssim_metric(img1, img2)
        write_result(f'SSIM (not verified): {ssim_res}', logger, out_file)

        psnr_res = psnr_metric(img1, img2)
        write_result(f'PSNR: {psnr_res}', logger, out_file)

        hdm_res = hamming_distance_metric(img1, img2)
        write_result(f'HDM: {hdm_res}', logger, out_file)

        csm_res = cosine_similarity_metric(img1, img2)
        write_result(f'CSM: {csm_res}', logger, out_file)
