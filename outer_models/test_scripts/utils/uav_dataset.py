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

    def __init__(self, root: str, verbose: bool = True, name_output: bool = True):
        """root -- путь до набора данных. Например, ./outer_models/test_scripts/dataset.
        verbose -- отображать ли некоторые данные."""

        self._root = root
        self._verbose = verbose
        self._name_output = name_output
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
        """Загрузка элемента."""

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

    def prepare_name(self, path: str):
        """Получение имени из пути."""

        name = os.path.split(path)[-1]
        return name

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

        if self._name_output:
            name = inst[0]
            if inst[2] == self.video_type:
                name += "_" + str(idx)
            return self.prepare_name(name), read_image
        else:
            return read_image


class UAVDataset(PreUAVDataset, IterableDataset):
    """Класс для набора данных с БПЛА с гетерогенным входом."""

    def __init__(self, root: str, verbose: bool = True, name_output=True):
        super().__init__(root, verbose, name_output)

    def generate(self):
        choice_structs = copy.deepcopy(self._data_struct)
        while choice_structs:
            current_length = len(choice_structs)
            this_length = random.randrange(current_length)
            inst = choice_structs.pop(this_length)
            if inst[2] == self.image_type:
                read_image = self.load_instance(inst, 0)
                if self._name_output:
                    yield self.prepare_name(inst[0]), read_image
                else:
                    yield read_image
            elif inst[2] == self.video_type:
                idx = 0
                while True:
                    cap = cv2.VideoCapture(inst[0])
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.moveaxis(frame, 2, 0)
                    read_image = torch.from_numpy(frame)
                    if self._name_output:
                        yield self.prepare_name(inst[0] + "_" + str(idx)), read_image
                    else:
                        yield read_image
                    idx += 1
            else:
                raise NotImplementedError("Incorrect file type:", inst[2], "!")

    def __iter__(self):
        return iter(self.generate())


if __name__ == "__main__":
    new_dataset = UAVDataset(r"./dataset")
    print(len(new_dataset))
    it = iter(new_dataset)
    b1, a1 = next(it)
    print(b1, a1.shape)
    b2, a2 = next(it)
    print(b2, a2.shape)
    b3, a3 = next(it)
    print(b3, a3.shape)
    b4, a4 = new_dataset[10000]
    print(b4, a4.shape)
