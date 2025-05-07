from omegaconf import OmegaConf
import torch
from dependence.util import instantiate_from_config
import os
import cv2
import numpy as np
import random
import math
import pytorch_lightning as pl
from skimage.metrics import structural_similarity
from utils.workers import WorkerASMoveDistribution, WorkerSRDummy

# from_dataset = "materials/dataset"
# dataset = "materials/dataset_addtrain1"
from_dataset = "dataset_preparation/spbsut_dataset_1000"
dataset = "dataset_preparation/spbsut_dataset_addtrain1"
dataset_type = torch.float32
nominal_type = torch.float16
worker_restore = WorkerASMoveDistribution()
worker_sr = WorkerSRDummy()

z_shape = (1, 3, 128, 128)
config_path = "dependence/config/kl-f4.yaml"
ckpt_path = "dependence/ckpt/kl-f4.ckpt"
config = OmegaConf.load(config_path)
pl_sd = torch.load(ckpt_path, map_location="cpu")  #
sd = pl_sd["state_dict"]
model_true = instantiate_from_config(config.model)
model_true.load_state_dict(sd, strict=False)
model_true.forward = model_true.decode
model_true = model_true.type(nominal_type).cuda()
model_true.eval()

model_test = torch.jit.load("kl-f4_decoder_cfg1_151000.ts")
model_test = model_test.type(nominal_type).cuda()
model_test.eval()


def mse_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики MSE.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = np.mean((image1 - image2) ** 2)

    return mse


def ssim_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики SSIM.
    На вход подаются две картинки в формате cv2 (numpy)."""

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

    return score


def psnr_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики PSNR.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


good_ext = [".lpp"]
def find_random_file(pather):
    catalogue = os.listdir(pather)
    new_node = random.sample(catalogue, 1)[0]
    new_path = os.path.join(pather, new_node)
    if os.path.isfile(new_path):
        if os.path.splitext(new_path)[1] in good_ext:
            return new_path
        else:
            return None
    else:
        return find_random_file(new_path)


lpp_filepath = None
while lpp_filepath is None:
    lpp_filepath = find_random_file(dataset)

rel_path = os.path.relpath(lpp_filepath, dataset)
old_path = os.path.splitext(os.path.join(from_dataset, rel_path))[0] + ".jpg"


with open(lpp_filepath, mode='rb') as lppf:
    compressed_bytes = lppf.read()
lpp_tensor = torch.frombuffer(compressed_bytes, dtype=dataset_type)
lpp_tensor = lpp_tensor.to(dtype=nominal_type)
lpp_tensor = lpp_tensor.reshape(z_shape)
lpp_tensor = lpp_tensor.to("cuda")


source_image = cv2.imread(old_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if source_image.shape[::-1][1:] != (1280, 720):
    image = cv2.resize(source_image, (1280, 720), interpolation=cv2.INTER_AREA)


def decoding(model, tens):
    image_pixel = model(tens)
    image_restore, _ = worker_restore.restore_work(image_pixel)
    image_restore, _ = worker_sr.sr_work(image_restore, dest_size=[1280, 720])

    return image_restore


true_model_img = decoding(model_true, lpp_tensor)
test_model_img = decoding(model_test, lpp_tensor)

ssim_true = ssim_metric(true_model_img, source_image)
mse_true = mse_metric(true_model_img, source_image)
psnr_true = psnr_metric(true_model_img, source_image)
ssim_test = ssim_metric(test_model_img, source_image)
mse_test = mse_metric(test_model_img, source_image)
psnr_test = psnr_metric(test_model_img, source_image)


print("--- Исходная модель ---")
print("SSIM = {:.02f}".format(ssim_true))
print("MSE = {:.04f}".format(mse_true))
print("PSNR = {:.04f} дБ".format(psnr_true))
print()
print("--- Дообученная модель ---")
print("SSIM = {:.02f}".format(ssim_test))
print("MSE = {:.04f}".format(mse_test))
print("PSNR = {:.04f} дБ".format(psnr_test))


cv2.imshow("SOURCE IMAGE", source_image)
cv2.imshow("TRUE MODEL", true_model_img)
cv2.imshow("TEST MODEL", test_model_img)
cv2.waitKey(0)

