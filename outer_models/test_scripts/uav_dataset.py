import os
import copy
import random
from functools import reduce
import numpy as np
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, IterableDataset


class PreUAVDataset(torch.utils.data.Dataset):
    """Заготовочный класс для набора данных с БПЛА с гетерогенным входом."""

    image_extensions = ("PNG", "JPG", "JPEG", "BMP")
    video_extensions = ("MP4", "AVI", "MOV")

    image_type = "img"
    video_type = "vid"

    def __init__(self, root: str, verbose: bool = True):
        """root -- путь до набора данных. Например, ./outer_models/test_scripts/dataset.
        verbose -- отображать ли некоторые данные."""

        self._root = root
        self._verbose = verbose
        self._data_struct = []
        #  [ [путь, длина, тип], ... ]

        for root, dirs, files in os.walk(root):
            if "__MACOSX" in root:
                continue

            for file in files:
                if "DS_Store" in file:
                    continue

                new_filepath = os.path.join(root, file)
                extension = os.path.splitext(new_filepath)[1][1:].upper()
                if extension in self.image_extensions:
                    length = 1
                    typer = self.image_type
                elif extension in self.video_extensions:
                    cap = cv2.VideoCapture(new_filepath)
                    length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    typer = self.video_type
                else:
                    if self._verbose:
                        print("Неподдерживаемое расширение:", new_filepath, "!")
                    continue

                new_struct = [new_filepath, length, typer]
                self._data_struct.append(new_struct)

        self._data_struct.sort(key=lambda x: x[1], reverse=True)

    def __len__(self):
        lenner = reduce(lambda a, b: ['', a[1]+b[1]], self._data_struct)[1]
        return lenner

    def load_instance(self, inst, idx=0):
        if inst[2] == self.image_type:
            read_image = torchvision.io.read_image(inst[0], mode=torchvision.io.ImageReadMode.RGB)
        elif inst[2] == self.video_type:
            cap = cv2.VideoCapture(inst[0])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, 2, 0)
            read_image = torch.from_numpy(frame)
        else:
            raise NotImplementedError("Incorrect file type:", inst[2], "!")
        return read_image

    def __getitem__(self, idx):
        global_idx = 0
        for structer in self._data_struct:
            if idx > structer[1]:
                idx -= structer[1]
                global_idx += 1
            else:
                break

        inst = self._data_struct[global_idx]
        read_image = self.load_instance(inst, idx)

        return read_image


class UAVDataset(PreUAVDataset, IterableDataset):
    def __init__(self, root: str, verbose: bool = True):
        super().__init__(root, verbose)

    def generate(self):
        choice_structs = copy.deepcopy(self._data_struct)
        while choice_structs:
            current_length = len(choice_structs)
            this_length = random.randrange(current_length)
            inst = choice_structs.pop(this_length)
            if inst[2] == self.image_type:
                yield self.load_instance(inst, 0)
            elif inst[2] == self.video_type:
                while True:
                    cap = cv2.VideoCapture(inst[0])
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.moveaxis(frame, 2, 0)
                    read_image = torch.from_numpy(frame)
                    yield read_image
            else:
                raise NotImplementedError("Incorrect file type:", inst[2], "!")

    def __iter__(self):
        return iter(self.generate())


if __name__ == "__main__":
    new_dataset = UAVDataset(r"./dataset")
    it = iter(new_dataset)
    a1 = next(it)
    print(a1.shape)
    a2 = next(it)
    print(a2.shape)
    a3 = next(it)
    print(a3.shape)
    a4 = new_dataset[10000]
    print(a4.shape)
