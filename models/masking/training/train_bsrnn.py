import os
import sys
import argparse

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import time
import torch
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from models.masking.bsrnn import BSRNN
from models.masking.masker import MaskMaster
from models.masking.training.latent_dataset import LatentDataset

arguments = argparse.ArgumentParser(prog="Обучение НСВБП (BSRNN)",
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
weights_saver = "models/masking/weights/bsrnn_{}_{}_ep{}"
os.makedirs(os.path.split(weights_saver)[0], exist_ok=True)

model = BSRNN(length=length, p=p)
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

current_epoch = 1

print("=== Начинаем обучение! ===")
while current_epoch <= max_epoch:
    epoch_start = time.time()

    # Обучение
    model.train()
    for latent in train_loader:
        float_latent = masker.prepare_latent(latent, typer=typer)
        mini_latent = masker.get_latent(float_latent).to(device=device)
        true_mask = masker.get_mask(float_latent).to(device=device)

        optim.zero_grad()

        eval_mask = model(mini_latent)
        loss_val = loss_fn(eval_mask, true_mask)
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

            float_latent = masker.prepare_latent(latent, typer=typer)
            mini_latent = masker.get_latent(float_latent).to(device=device)
            true_mask = masker.get_mask(float_latent).to(device=device)

            optim.zero_grad()

            eval_mask = model(mini_latent)
            loss_val = loss_fn(eval_mask, true_mask)
            loss_val = torch.sqrt(loss_val)
            loss_val = loss_val.item()

            total_num += batch_size
            total_loss += loss_val
    mean_loss = total_loss / total_num

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print("=== Эпоха {} ({:.4f} с): RMSEср = {:f}".format(current_epoch, epoch_time, mean_loss))

    current_epoch += 1
print("=== Обучение успешно завершено! ===")
