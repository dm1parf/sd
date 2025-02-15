import os
import copy
import random
from functools import reduce
import numpy as np
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, random_split


class LatentDataset(torch.utils.data.Dataset):
    """Заготовочный класс для латентного набора данных."""

    extensions = ("LAT", "LATENT")

    def __init__(self, root: str, verbose: bool = True, name_output: bool = True):
        """root -- путь до набора данных. Например, ./dependence/materials/dataset.
        verbose -- отображать ли некоторые данные."""

        self._root = root
        self._verbose = verbose
        self._name_output = name_output
        self._data_struct = []
        #  [filepath, filepath, ...]

        for root, dirs, files in os.walk(root):
            for file in files:
                new_filepath = os.path.join(root, file)
                extension = os.path.splitext(new_filepath)[1][1:].upper()
                if extension not in self.extensions:
                    if self._verbose:
                        print("Неподдерживаемое расширение:", new_filepath, "!")
                    continue
                self._data_struct.append(new_filepath)

        self._current_length = len(self._data_struct)
        self._generator = torch.Generator().manual_seed(33)

    def __len__(self):
        return self._current_length

    def prepare_name(self, path: str):
        """Получение имени из пути."""

        name = os.path.split(path)[-1]
        return name

    def load_instance(self, inst):
        """Чтение файла латента."""

        with open(inst, mode='rb') as rf:
            compressed_bytes = rf.read()
        latent = torch.frombuffer(compressed_bytes, dtype=torch.uint8)
        return latent

    def get_train_test(self, test_rate: float) -> (Dataset, Dataset):
        """Разделение набора."""

        train_rate = 1.0 - test_rate
        train_dataset, test_dataset = random_split(self, (train_rate, test_rate), self._generator)

        return train_dataset, test_dataset

    def __getitem__(self, idx):
        if isinstance(idx, list):
            all_list = []
            for ind in idx:
                inst = self._data_struct[ind]
                latent = self.load_instance(inst)
                all_list.append(latent)
            all_latent = torch.cat(all_list, dim=0)
            return all_latent
        else:
            inst = self._data_struct[idx]
            latent = self.load_instance(inst)

            if self._name_output:
                name = self.prepare_name(inst)
                return name, latent
            else:
                return latent


def test():
    import os
    import sys
    cwd = os.getcwd()  # Linux fix
    if cwd not in sys.path:
        sys.path.append(cwd)
    from torch.utils.data import RandomSampler, BatchSampler, DataLoader
    from models.masking.masker import MaskMaster

    dataset_dir = "materials/dataset_kl-f14_latent"
    test_rate = 0.2
    batch_size = 2
    length = 16_384
    p = 0.05
    device = "cuda"
    typer = torch.float32

    masker = MaskMaster(length=length, p=p)

    dataset = LatentDataset(dataset_dir, verbose=False, name_output=False)
    train_dataset, test_dataset = dataset.get_train_test(test_rate)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    lim = 5
    i = 0
    print("1", len(train_loader))
    for latent in train_loader:
        print("-", latent.shape, latent.dtype)
        print(latent[0:10])
        float_latent = masker.prepare_latent(latent, typer=typer)
        print("--", float_latent.shape, float_latent.dtype)
        print(float_latent[0:10])

        i += 1
        if i > lim:
            break

    print("2", len(test_loader))
    for latent in test_loader:
        print("-", latent.shape, latent.dtype)
        print(latent[0:10])
        float_latent = masker.prepare_latent(latent, typer=typer)
        print("--", float_latent.shape, float_latent.dtype)
        print(float_latent[0:10])

        i += 1
        if i > lim:
            break


if __name__ == "__main__":
    test()
