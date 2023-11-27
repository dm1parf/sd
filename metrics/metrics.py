import numpy as np
from scipy.spatial.distance import hamming
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import Compose, Resize, ToTensor
import lpips
# import ffmpeg_quality_metrics as ffqm
import erqa
import cv2
from pytorch_msssim import MS_SSIM
import math
import torch
from constants.constant import DEVICE

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
lpips_model.eval()


ms_ssim_module = MS_SSIM(data_range=255, size_average=False, channel=3)
ms_ssim_module = ms_ssim_module.to(DEVICE)
ms_ssim_module.eval()


def cosine_similarity_metric(image1, image2):
    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
            [image1_vector], [image2_vector])[0][0]

    return similarity_score


def hamming_distance_metric(image1, image2):
    return hamming(image1.flatten(), image2.flatten())


def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse


def ssim(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image1, image2, data_range=image2.max() - image2.min())

    return score


def yuv_ssim_metric(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = structural_similarity(
        image1, image2, data_range=(image2.max() - image2.min()))

    return score


def cor_pirson(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    corr = ((data1 - mean1) * (data2 - mean2)).mean() / (std1 * std2)
    return corr


def lpips_metric(image1, image2):
    train_transforms = Compose([
        Resize((512, 512)),
        ToTensor()
    ])
    image1 = train_transforms(image1).to(DEVICE)
    image2 = train_transforms(image2).to(DEVICE)

    with torch.no_grad():
        metrics = lpips_model.forward(image1, image2).cpu()
    return 1 - metrics.item()


# def vmaf(image1, image2):
#     vmaf_score = ffqm.FfmpegQualityMetrics(image1, image2)
#     metrics = vmaf_score.calculate(["vmaf"])
#     return [metrics['vmaf']]


def erqa_metrics(image1, image2):
    metric = erqa.ERQA()
    result = metric(image1, image2)
    return result


def msssim(image1, image2):
    with torch.no_grad():
        image1 = image1.transpose(2, 0, 1)
        image2 = image2.transpose(2, 0, 1)

        image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        score = ms_ssim_module(image1, image2).cpu().squeeze().numpy()

    return float(score)


def psnr_metric(image1, image2):
    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def yuv_psnr_metric(image1, image2):
    image1_y_component = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)[:, :, 0]
    image2_y_component = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)[:, :, 0]

    score = psnr_metric(image1_y_component, image2_y_component)

    return score