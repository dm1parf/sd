import os
import sys
import argparse

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import time
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from models.masking.regression import LN_RPM
from models.masking.masker import MaskMaster
from models.masking.training.latent_dataset import LatentDataset

arguments = argparse.ArgumentParser(prog="Обучение ЛнРПМ (LnRPV)",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-p", dest="p", type=float, default=0.05, help="Значение p")
arguments.add_argument("-l", dest="length", type=int, default=16_384, help="Значение length")
args = arguments.parse_args()

length = args.length
p = args.p
typer = np.float32

batch_size = 1000
max_epoch = 1
dataset_dir = "materials/dataset_kl-f14_latent"
test_rate = 0.2
weights_saver = "models/masking/ln_rpm_weights/ln_rpm_{}_{:.2f}"
os.makedirs(os.path.split(weights_saver)[0], exist_ok=True)

all_p = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
         0.95]


def rmse_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики MSE.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = np.sqrt(np.mean((image1 - image2) ** 2))

    return mse


# for p in all_p:
if True:
    print("=== Начинаем обучение: p = {} ===".format(p))

    model = LN_RPM(length=length, p=p, bandwidth=5)
    masker = MaskMaster(length=length, p=p)

    dataset = LatentDataset(dataset_dir, verbose=False, name_output=False)
    train_dataset, test_dataset = dataset.get_train_test(test_rate)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    last_rmse = 100500
    current_epoch = 1
    while current_epoch <= max_epoch:
        epoch_start = time.time()

        # Обучение
        for latent in train_loader:
            float_latent = latent.numpy()
            mini_latent = masker.get_latent(float_latent)
            true_mask = masker.get_mask(float_latent)

            model.train(mini_latent, true_mask)
            break

        weight_path = weights_saver.format(length, p, current_epoch)
        model.save(weight_path)

        # Проверка
        total_num = 0
        total_loss = 0.0

        for latent in test_loader:
            batch_sizer = latent.shape[0]

            for i in range(batch_sizer):
                float_latent = latent[i].reshape(1, -1).numpy()
                mini_latent = masker.get_latent(float_latent)
                true_mask = masker.get_mask(float_latent)

                eval_mask = model.eval(mini_latent)
                loss_val = rmse_metric(eval_mask, true_mask)

                total_num += 1
                total_loss += loss_val
            break

        mean_loss = total_loss / total_num

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        not_big_loss = round(mean_loss, 6)

        print("> Эпоха {} ({:.4f} с): RMSEср = {:f}".format(current_epoch, epoch_time, mean_loss))
        if not_big_loss >= last_rmse:
            break
        else:
            last_rmse = not_big_loss

        current_epoch += 1
print("=== Обучение успешно завершено! ===")
