import numpy as np
from scipy.spatial.distance import hamming
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import Compose, Resize, ToTensor
import lpips
import ffmpeg_quality_metrics as ffqm
import erqa
import cv2
from sewar.full_ref import msssim
import math
import torch
from constants.constant import DEVICE
import time
import logging
import functools

logger = logging.getLogger('main')

# lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
lpips_model = None

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer
def cosine_similarity_metric(image1, image2):
    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
            [image1_vector], [image2_vector])[0][0]

    return similarity_score

@timer
def hamming_distance_metric(image1, image2):
    return hamming(image1.flatten(), image2.flatten())

@timer
def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse

@timer
def ssim(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image1, image2, data_range=image2.max() - image2.min())

    return score

@timer
def yuv_ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = structural_similarity(
        image1, image2, data_range=(image2.max() - image2.min()))

    return score

@timer
def cor_pirson(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    corr = ((data1 - mean1) * (data2 - mean2)).mean() / (std1 * std2)
    return corr

@timer
def lpips_metric(image1, image2):
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
    train_transforms = Compose([
        Resize((512, 512)),
        ToTensor()
    ])
    image1 = train_transforms(image1).to(DEVICE)
    image2 = train_transforms(image2).to(DEVICE)

    lpips_model.eval()
    with torch.no_grad():
        metrics = lpips_model.forward(image1, image2).cpu()
    return 1 - metrics.item()

@timer
def vmaf(image1, image2):
    vmaf_score = ffqm.FfmpegQualityMetrics(image1, image2)
    metrics = vmaf_score.calculate(["vmaf"])
    return [metrics['vmaf']]

@timer
def erqa_metrics(image1, image2):
    metric = erqa.ERQA()
    result = metric(image1, image2)
    return result

@timer
def ms_ssim(image1, image2):
    image1_y_component = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2_y_component = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]
    result = msssim(image1_y_component, image2_y_component)
    return result

@timer
def psnr_metric(image1, image2):
    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr

@timer
def yuv_psnr_metric(image1, image2):
    image1_y_component = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2_y_component = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = psnr_metric(image1_y_component, image2_y_component)

    return score