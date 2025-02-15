import os
import sys
import argparse

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import time
import torch
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from models.masking.simple_mask_nn import SBSRNN_F
from models.masking.masker import MaskMaster
from models.masking.training.latent_dataset import LatentDataset

arguments = argparse.ArgumentParser(prog="Обучение ПНСВБП-П (SBSRNN-F)",
                                    description="Сделано для испытаний канала.")
arguments.add_argument("-p", dest="p", type=float, default=0.05, help="Значение p")
arguments.add_argument("-l", dest="length", type=int, default=16_384, help="Значение length")
args = arguments.parse_args()

length = args.length
p = args.p
device = "cuda"
typer = torch.float32

max_epoch = 10
batch_size = 1000
learning_rate = 0.001
dataset_dir = "materials/dataset_kl-f14_latent"
test_rate = 0.2
weights_saver = "models/masking/weights/sbsrnn_f_{}_{}_ep{}"
os.makedirs(os.path.split(weights_saver)[0], exist_ok=True)

all_p = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
         0.95]

# for p in all_p:
if True:
    print("=== Начинаем обучение: p = {} ===".format(p))

    model = SBSRNN_F(length=length, p=p)
    model = model.to(dtype=typer, device=device)
    masker = MaskMaster(length=length, p=p)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

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
        model.train()
        for latent in train_loader:
            optim.zero_grad()

            float_latent = masker.prepare_latent(latent, typer=typer).to(device=device)
            mini_latent = masker.get_latent(float_latent)
            eval_latent = model(mini_latent)
            loss_val = loss_fn(eval_latent, float_latent)
            loss_val = torch.sqrt(loss_val)
            loss_val.backward()

            optim.step()

        weight_path = weights_saver.format(length, p, current_epoch)
        torch.save(model.state_dict(), weight_path)

        # Проверка
        model.eval()
        total_num = 0
        total_loss = 0.0
        with torch.no_grad():
            for latent in test_loader:
                batch_size = latent.shape[0]

                optim.zero_grad()

                float_latent = masker.prepare_latent(latent, typer=typer).to(device=device)
                mini_latent = masker.get_latent(float_latent)
                eval_latent = model(mini_latent)
                loss_val = loss_fn(eval_latent, float_latent)
                loss_val = torch.sqrt(loss_val)
                loss_val = loss_val.item()

                total_num += batch_size
                total_loss += loss_val
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
