import argparse
from models.restoration.ncirnn import NCIRNN
from utils.workers import WorkerSRDummy, WorkerASMoveDistribution
import torch
import copy
import numpy as np
import cv2
from skimage.metrics import structural_similarity
import math
import os
import time
import random
# import lpips


arguments = argparse.ArgumentParser(prog="Обучение NCIRNN",
                                    description="Восстановление изображений.")
arguments.add_argument("-c", dest="cfg", type=int, default=1, help="Конфигурация")
args = arguments.parse_args()
cfg = args.cfg

# Test NN structure.
train_dataset_source = "materials/dataset"
train_dataset_cfg = "materials/dataset_cfg{}".format(cfg)
test_dataset_source = "dataset_preparation/spbsut_25fps_full"
test_dataset_cfg = "dataset_preparation/cfg{}_spbsut_25_full".format(cfg)
# train_dataset_source = test_dataset_source
# train_dataset_cfg = test_dataset_cfg 
abspath_test = os.path.abspath(test_dataset_source)
abspath_train = os.path.abspath(train_dataset_source)
source_ext = [".jpg", ".bmp", ".png"]
filewrite = "models/restoration/ncirnn_train_log.txt"
fw = open(filewrite, encoding="utf-8", mode='w')
model_filer = "ncirnn_cfg1_{}.ts"

model = NCIRNN()

nominal_type = torch.float32
learning_rate = 0.001
train_rand_threshold = 1  # 0.01 1
train_maxval = 300000
test_every = 10000  # 1000
recurr_dataset = False  # True False

model.to(dtype=nominal_type, device="cuda")

loss_fn = torch.nn.MSELoss()
# loss_fn = lpips.LPIPS(net='alex')
# loss_fn.to(device="cuda", dtype=nominal_type)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

worker_sr = WorkerSRDummy()
worker_restore = WorkerASMoveDistribution()


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


def load_train_image(path):
    from_img = cv2.imread(path)
    torch_img, _ = worker_restore.prepare_work(from_img, dest_type=nominal_type)

    return torch_img


last_best = 999_999_999
last_definer = -1
def test_model(name_definer):
    global last_best
    global last_definer

    model_clone = copy.deepcopy(model)
    model_clone.to(dtype=torch.float16)
    model_clone.eval()

    total_ssim = 0
    total_mse = 0
    total_psnr = 0
    test_counter = 0
    total_time_ms = 0
    with torch.no_grad():
        for root, dirs, files in os.walk(test_dataset_source):
            for file in files:
                this_full_file = os.path.join(root, file)
                relfile = os.path.relpath(this_full_file, abspath_test)

                base_file, extension = os.path.splitext(relfile)
                if extension.lower() not in source_ext:
                    continue

                source_file = os.path.join(test_dataset_source, relfile)
                cfg_file = os.path.join(test_dataset_cfg, relfile)

                true_image = cv2.imread(source_file)
                cfg_image = load_train_image(cfg_file).to(dtype=torch.float16)
                a = time.time()
                image_predict = model_clone(cfg_image)
                b = time.time()
                image_restore, _ = worker_restore.restore_work(image_predict)
                image_restore, _ = worker_sr.sr_work(image_restore, dest_size=[1280, 720])

                ssim = ssim_metric(image_restore, true_image)
                mse = mse_metric(image_restore, true_image)
                psnr = psnr_metric(image_restore, true_image)

                time_ms = (b - a) * 1000

                total_time_ms += time_ms
                total_ssim += ssim
                total_mse += mse
                total_psnr += psnr

                test_counter += 1
    mean_ssim = total_ssim / test_counter
    mean_mse = total_mse / test_counter
    mean_psnr = total_psnr / test_counter
    mean_ms = total_time_ms / test_counter
    print("--- Тестирование: {} ---".format(name_definer), file=fw)
    print("SSIMср = {:.2f}".format(mean_ssim), file=fw)
    print("MSEср = {:.2f}".format(mean_mse), file=fw)
    print("PSNRср = {:.2f}".format(mean_psnr), file=fw)
    print("Timemsср = {:.2f}".format(mean_ms), file=fw)
    print("--- Тестирование: {} ---".format(name_definer))
    print("SSIMср = {:.2f}".format(mean_ssim))
    print("MSEср = {:.2f}".format(mean_mse))
    print("PSNRср = {:.2f}".format(mean_psnr))
    print("Timemsср = {:.2f}".format(mean_ms))

    if mean_mse < last_best:
        last_best = mean_mse
        last_file = model_filer.format(last_definer)
        if last_definer != -1:
            os.remove(last_file)
        last_definer = name_definer
        model_save_file = model_filer.format(name_definer)
        inp = [torch.randn(1, 3, 512, 512, dtype=nominal_type, device='cuda')]
        traced_model = torch.jit.trace(model, inp)
        torch.jit.save(traced_model, model_save_file)


train_counter = 0
break_flag = False
dest_ext = ".lpp"
last_test = -1
model.train()
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
            if extension.lower() not in source_ext:
                continue

            source_file = os.path.join(train_dataset_source, relfile)
            cfg_file = os.path.join(train_dataset_cfg, relfile)

            with torch.no_grad():
                true_img = load_train_image(source_file)
                cfg_image = load_train_image(cfg_file)

            optim.zero_grad()
            image_predict = model(cfg_image)
            loss_val = loss_fn(image_predict, true_img)  # 512x512 - 512x512

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



