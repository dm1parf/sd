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

# Скрипт для дообучения

train_dataset_source = "materials/dataset"
train_dataset_lpp = "materials/dataset_addtrain1"
test_dataset_source = "dataset_preparation/spbsut_dataset_1000"
test_dataset_lpp = "dataset_preparation/spbsut_dataset_addtrain1"
model_filer = "kl-f4_decoder_cfg1_{}.ts"
filewrite = "addtrain1_log.txt"
fw = open(filewrite, encoding="utf-8", mode='w')

config_path = "dependence/config/kl-f4.yaml"
ckpt_path = "dependence/ckpt/kl-f4.ckpt"
nominal_type = torch.float32
train_maxval = 300_000
# 10_000
test_every = 1_000 # train_maxval // 10
learning_rate = 1.0e-06
z_shape = (1, 3, 128, 128)
train_rand_threshold = 1  # 0.01
is_true_source = True
recurr_dataset = False  # True

if is_true_source:
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")  #
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.forward = model.decode
    model._trainer = pl.Trainer()
else:
    model = torch.jit.load("kl-f4_decoder_cfg1_0.ts")
model = model.type(nominal_type).cuda()
model.train()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))

loss_fn = torch.nn.MSELoss()
worker_restore = WorkerASMoveDistribution()
worker_sr = WorkerSRDummy()


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


def prepare_image_lpp(source_file, lpp_file):
    image = cv2.imread(source_file)
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # TODO check
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = np.moveaxis(image, 2, 0)
    image = torch.from_numpy(image)
    image = image.cuda()
    image = image.to(nominal_type)
    current_shape = image.shape
    image = image / 255.0
    image = image.reshape(1, *current_shape)
    """
    image, _ = worker_restore.prepare_work(image, dest_type=nominal_type)

    with open(lpp_file, mode='rb') as lppf:
        compressed_bytes = lppf.read()
    lpp_tensor = torch.frombuffer(compressed_bytes, dtype=nominal_type)
    lpp_tensor = lpp_tensor.reshape(z_shape)
    lpp_tensor = lpp_tensor.to("cuda")

    return image, lpp_tensor


def prepare_image_lpp_test(source_file, lpp_file):
    image = cv2.imread(source_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[::-1][1:] != (1280, 720):
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

    with open(lpp_file, mode='rb') as lppf:
        compressed_bytes = lppf.read()
    lpp_tensor = torch.frombuffer(compressed_bytes, dtype=nominal_type)
    lpp_tensor = lpp_tensor.reshape(z_shape)
    lpp_tensor = lpp_tensor.to("cuda")

    return image, lpp_tensor


abspath_test = os.path.abspath(test_dataset_source)


def test_model(name_definer):
    global model

    model.eval()
    total_ssim = 0
    total_mse = 0
    total_psnr = 0
    test_counter = 0
    with torch.no_grad():
        for root, dirs, files in os.walk(test_dataset_source):
            for file in files:
                this_full_file = os.path.join(root, file)
                relfile = os.path.relpath(this_full_file, abspath_test)

                base_file, extension = os.path.splitext(relfile)
                if extension != source_ext:
                    continue

                source_file = os.path.join(test_dataset_source, base_file + source_ext)
                lpp_file = os.path.join(test_dataset_lpp, base_file + dest_ext)

                image, lpp_tensor = prepare_image_lpp_test(source_file, lpp_file)

                image_predict = model(lpp_tensor)
                image_restore, _ = worker_restore.restore_work(image_predict)
                image_restore, _ = worker_sr.sr_work(image_restore, dest_size=[1280, 720])

                ssim = ssim_metric(image_restore, image)
                mse = mse_metric(image_restore, image)
                psnr = psnr_metric(image_restore, image)

                total_ssim += ssim
                total_mse += mse
                total_psnr += psnr

                test_counter += 1
    mean_ssim = total_ssim / test_counter
    mean_mse = total_mse / test_counter
    mean_psnr = total_psnr / test_counter
    print("--- Тестирование: {} ---".format(name_definer), file=fw)
    print("SSIMср = {:.2f}".format(mean_ssim), file=fw)
    print("MSEср = {:.2f}".format(mean_mse), file=fw)
    print("PSNRср = {:.2f}".format(mean_psnr), file=fw)
    print("--- Тестирование: {} ---".format(name_definer))
    print("SSIMср = {:.2f}".format(mean_ssim))
    print("MSEср = {:.2f}".format(mean_mse))
    print("PSNRср = {:.2f}".format(mean_psnr))

    model_save_file = model_filer.format(name_definer)
    if is_true_source:
        # AutoencoderKL has no attribute save.
        inp = [torch.randn(1, 3, 128, 128, dtype=nominal_type, device='cuda')]
        traced_model = torch.jit.trace(model, inp)
        torch.jit.save(traced_model, model_save_file)
    else:
        torch.jit.save(model, model_save_file)

    model.train()


abspath_train = os.path.abspath(train_dataset_source)
train_counter = 0
break_flag = False
source_ext = ".jpg"
dest_ext = ".lpp"
last_test = -1

while train_counter < train_maxval:
    for root, dirs, files in os.walk(train_dataset_source):
        for file in files:
            if (train_counter % test_every) == 0:
                # Проверка
                if last_test != train_counter:
                    test_model(train_counter)
                    last_test = train_counter

            nrant = random.random()
            if nrant > train_rand_threshold:
                continue

            this_full_file = os.path.join(root, file)
            relfile = os.path.relpath(this_full_file, abspath_train)

            base_file, extension = os.path.splitext(relfile)
            if extension != source_ext:
                continue

            source_file = os.path.join(train_dataset_source, base_file + source_ext)
            lpp_file = os.path.join(train_dataset_lpp, base_file + dest_ext)

            image, lpp_tensor = prepare_image_lpp(source_file, lpp_file)

            optim.zero_grad()
            image_predict = model(lpp_tensor)
            a, _ = worker_restore.restore_work(image_predict)
            b, _ = worker_restore.restore_work(image)
            loss_val = loss_fn(image_predict, image)  # 512x512 - 512x512

            loss_val.backward()
            optim.step()

            train_counter += 1
            if train_maxval and train_counter >= train_maxval:
                break_flag = True
                break

        if break_flag:
            break
    if not recurr_dataset:
        break

test_model(train_counter)

fw.close()
