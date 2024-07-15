import copy
import math
import time
import zlib
import lzma
import bz2
import gzip
import io
import os
from functools import reduce
from typing import Callable, Union
from abc import abstractmethod

# import av.error
import cv2
import numpy
import torch
import torchvision
import imageio
import imagecodecs
import pillow_avif
import pillow_heif
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as tvfunc
from omegaconf import OmegaConf
from basicsr.archs.rrdbnet_arch import RRDBNet
from dependence.util import instantiate_from_config
from dependence.realesrgan import RealESRGANer
from dependence.prediction.model.models import Model as Predictor, DMVFN
from dependence.cdc.compress_modules import BigCompressor
from dependence.cdc.denoising_diffusion import GaussianDiffusion
from dependence.cdc.unet import Unet
from dependence.apisr.test_utils import load_grl, load_rrdb


# WorkerMeta -- метакласс для декорации -> получения времени
# WorkerDummy -- класс ложного ("ленивого") рабочего, имитирующего деятельность

# > WorkerASInterface -- абстрактный класс подавителя артефактов
# WorkerASDummy -- класс ленивого подавителя артефактов, просто делает переводы форматов и ничего не подавляет
# WorkerASCutEdgeColors -- класс подавителя артефактов, что обрезает цвета
# WorkerASMoveDistribution -- класс подавителя артефактов, что переносит распределение
# WorkerASComposit -- класс подавителя артефактов, что переносит распределение

# > WorkerAutoencoderInterface -- абстрактный класс интерфейса для автокодировщиков
# WorkerAutoencoderVQ_F4 -- класс рабочего вариационного автокодировщика VQ-f4
# WorkerAutoencoderVQ_F8 -- класс рабочего вариационного автокодировщика VQ-f8
# WorkerAutoencoderVQ_F16 -- класс рабочего вариационного автокодировщика VQ-f16
# WorkerAutoencoderVQ_F16_Optimized -- класс рабочего оптимизированного вариационного автокодировщика VQ-f16
# WorkerAutoencoderKL_F4 -- класс рабочего вариационного автокодировщика KL-f4
# WorkerAutoencoderKL_F8 -- класс рабочего вариационного автокодировщика KL-f8
# WorkerAutoencoderKL_F16 -- класс рабочего вариационного автокодировщика KL-f16
# WorkerAutoencoderKL_F32 -- класс рабочего вариационного автокодировщика KL-f32
# WorkerAutoencoderCDC -- класс рабочего вариационного автокодировщика CDC

# > WorkerQuantInterface -- абстрактный класс интерфейса для квантования
# WorkerQuantLinear -- класс рабочего для линейного квантования и деквантования
# WorkerQuantPower -- класс рабочего для степенного квантования и деквантования
# WorkerQuantLogistics -- класс рабочего для логистического квантования и деквантования
# WorkerQuantMinLogistics -- класс рабочего для модифицированного логистического квантования и деквантования
# WorkerQuantOddPower -- класс рабочего для нечётностепенного квантования и деквантования
# WorkerQuantTanh -- класс рабочего для гиперболическотангенциального квантования и деквантования
# WorkerQuantMinTanh -- класс рабочего для модифицированного (mean -> min) гиперболическотангенциального квантования и деквантования
# WorkerQuantDoubleLogistics -- класс рабочего для двойного логистического квантования и деквантования
# WorkerQuantMinDoubleLogistics -- класс рабочего для модифицированного двойного логистического квантования и деквантования
# WorkerQuantSinh -- класс рабочего для гиперболическисинусоидального квантования и деквантования

# > WorkerCompressorInterface -- абстрактный класс интерфейса для сжатия/расжатия
# WorkerCompressorDummy -- класс ложного ("ленивого") рабочего, имитирующего сжатие
# WorkerCompressorDeflated -- класс рабочего для сжатия и расжатия Deflated
# WorkerCompressorLzma -- класс рабочего для сжатия и расжатия Lzma
# WorkerCompressorGzip -- класс рабочего для сжатия и расжатия Gzip
# WorkerCompressorBzip2 -- класс рабочего для сжатия и расжатия Bzip2
# WorkerCompressorZstd -- класс рабочего для сжатия и расжатия ZSTD (ZStandard)
# WorkerCompressorBrotli -- класс рабочего для сжатия и расжатия Brotli
# WorkerCompressorLz4 -- класс рабочего для сжатия и расжатия LZ4
# WorkerCompressorLz4f -- класс рабочего для сжатия и расжатия LZ4F
# WorkerCompressorLz4h5 -- класс рабочего для сжатия и расжатия LZ4H5
# WorkerCompressorLzw -- класс рабочего для сжатия и расжатия LZW
# WorkerCompressorLzf -- класс рабочего для сжатия и расжатия LZF
# WorkerCompressorLzfse -- класс рабочего для сжатия и расжатия LZF
# WorkerCompressorH264 -- класс рабочего для сжатия и расжатия H264
# WorkerCompressorH265 -- класс рабочего для сжатия и расжатия H265
# WorkerCompressorJpeg -- класс рабочего для сжатия и расжатия JPEG
# WorkerCompressorAvif -- класс рабочего для сжатия и расжатия AVIF
# WorkerCompressorHeic -- класс рабочего для сжатия и расжатия HEIC
# WorkerCompressorWebp -- класс рабочего для сжатия и расжатия WebP
# WorkerCompressorJpegLS -- класс рабочего для сжатия и расжатия JPEG LS
# WorkerCompressorJpegXR -- класс рабочего для сжатия и расжатия JPEG XR
# WorkerCompressorJpegXL -- класс рабочего для сжатия и расжатия JPEG XL
# WorkerCompressorQoi -- класс рабочего для сжатия и расжатия QOI

# > WorkerSRInterface -- абстрактный класс интерфейса для суперрезолюции
# WorkerSRDummy -- класс ложного ("ленивого") рабочего, имитирующего суперрезолюцию
# WorkerSRRealESRGAN_x2plus -- класс рабочего SR вида ESRGAN Plus x2
# WorkerSRAPISR_RRDB_x2 -- класс рабочего SR вида APISR x2 (основан на ESRGAN)
# WorkerSRAPISR_RRDB_x2_Optimized -- класс рабочего SR вида APISR x2 (основан на ESRGAN)
# WorkerSRAPISR_GRL_x4 -- класс рабочего SR вида GRL x4 (tiny2)

# > WorkerPredictorInterface -- абстрактный класс интерфейса для предиктора
# WorkerPredictorDummy -- класс ложного ("ленивого") рабочего, имитирующего предиктор
# WorkerPredictorDMVFN -- класс рабочего предиктора на основе DMVFN


class WorkerMeta(type):
    """Метакласс для т.н. рабочих объектов -- строительных блоков системы экспериментов.
    Все методы, кончающиеся на _work, декорируются и вторым аргументом возвращают время.
    Также добавляет аргумент strict_sync для строгой синхронизации (torch.cuda.syncronize())."""

    work_method_start = "_work"

    def __new__(cls, name, parents, attrdict):
        for key in attrdict:
            if key.endswith("_work") and isinstance(attrdict[key], Callable):
                attrdict[key] = WorkerMeta.time_decorator(attrdict[key])
        return type.__new__(cls, name, parents, attrdict)

    @staticmethod
    def time_decorator(func: Callable) -> Callable:
        def internal_func(*args, strict_sync: bool = False, milliseconds_mode: bool = False, **kwargs):
            if strict_sync:
                torch.cuda.synchronize()
            if milliseconds_mode:
                start = time.time_ns()
            else:
                start = time.time()
            result = func(*args, **kwargs)
            if strict_sync:
                torch.cuda.synchronize()
            if milliseconds_mode:
                end = time.time_ns()
                delta = (end - start) // 1_000_000
            else:
                end = time.time()
                delta = end - start
            return result, delta

        return internal_func


class WorkerDummy(metaclass=WorkerMeta):
    """Ложный рабочий."""

    def __init__(self, wait=0.5, verbose=True):
        self.wait = wait
        self._verbose = verbose

    def do_work(self, *_, **__):
        time.sleep(self.wait)
        if self._verbose:
            print("...")


class WorkerASInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих-подавителей артефактов."""

    @abstractmethod
    def prepare_work(self, from_image: np.ndarray, dest_type=torch.float16) -> torch.Tensor:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = cv2.cvtColor(from_image, cv2.COLOR_BGR2RGB)
        if image.shape[::-1][1:] != (self.middle_width, self.middle_height):
            image = cv2.resize(image, (self.middle_width, self.middle_height), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image)
        image = image.cuda()

        image = image.to(dest_type)
        current_shape = image.shape
        image = image / 255.0
        image = image.reshape(1, *current_shape)

        return image

    @abstractmethod
    def restore_work(self, from_image: torch.Tensor) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        from_image *= 255.0
        image = from_image.to(torch.uint8)
        
        image = image.reshape(3, self.middle_height, self.middle_width)

        image = image.cpu()
        end_numpy = image.numpy()
        end_numpy = np.moveaxis(end_numpy, 0, 2)
        end_numpy = cv2.cvtColor(end_numpy, cv2.COLOR_BGR2RGB)

        return end_numpy


class WorkerASDummy(WorkerASInterface):
    """Интерфейс для рабочих-подавителей артефактов."""

    def __init__(self, middle_width: int = 512, middle_height: int = 512):
        """
        :param middle_width: Промежуточная ширина.
        :param middle_height: Промежуточная высота.
        """

        self.middle_width = int(middle_width)
        self.middle_height = int(middle_height)

    def prepare_work(self, from_image: np.ndarray, dest_type=torch.float16) -> torch.Tensor:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image, _ = super().prepare_work(from_image, dest_type)

        return image

    def restore_work(self, from_image: torch.Tensor) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        end_numpy, _ = super().restore_work(from_image)

        return end_numpy


class WorkerASCutEdgeColors(WorkerASInterface):
    """Рабочий-подавитель артефактов посредством убирания крайних цветов."""

    def __init__(self, delta: Union[str, int, float] = 15, middle_width: int = 512, middle_height: int = 512):
        """delta -- насколько крайние цвета в RGB.
        middle_width -- ширина промежуточной картинки.
        middle_height -- высота промежуточной картинки."""

        self.middle_width = int(middle_width)
        self.middle_height = int(middle_height)

        self._delta = int(delta)
        self._low = self._delta
        self._high = 255 - self._delta
        self._low_array = np.array([self._low, self._low, self._low])
        self._high_array = np.array([self._high, self._high, self._high])

    def prepare_work(self, from_image: np.ndarray, dest_type=torch.float16) -> torch.Tensor:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = from_image
        low_mask = np.sum(image < self._low, axis=2) == 3
        high_mask = np.sum(image > self._high, axis=2) == 3

        image[low_mask] = self._low_array
        image[high_mask] = self._high_array

        image, _ = super().prepare_work(image, dest_type)

        return image

    def restore_work(self, from_image: torch.Tensor) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        end_numpy, _ = super().restore_work(from_image)

        return end_numpy


class WorkerASMoveDistribution(WorkerASInterface):
    """Рабочий-подавитель артефактов посредством убирания крайних цветов."""

    def __init__(self, middle_width: int = 512, middle_height: int = 512):
        """
        :param middle_width: Промежуточная ширина.
        :param middle_height: Промежуточная высота.
        """

        self.middle_width = int(middle_width)
        self.middle_height = int(middle_height)

    def prepare_work(self, from_image: np.ndarray, dest_type=torch.float16) -> torch.Tensor:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image, _ = super().prepare_work(from_image, dest_type)
        image = image * 2.0 - 1.0

        return image

    def restore_work(self, from_image: torch.Tensor) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        image = (from_image / 2 + 0.5).clamp(0, 1)
        end_numpy, _ = super().restore_work(image)

        return end_numpy


class WorkerASComposit(WorkerASInterface):
    """Рабочий-подавитель артефактов посредством убирания крайних цветов."""

    def __init__(self, delta: Union[str, int, float] = 15, middle_width: int = 512, middle_height: int = 512):
        """
        :param delta: размер крайних цветов.
        :param middle_width: Промежуточная ширина.
        :param middle_height: Промежуточная высота.
        """

        self.middle_width = int(middle_width)
        self.middle_height = int(middle_height)

        self._delta = int(delta)
        self._low = self._delta
        self._high = 255 - self._delta
        self._low_array = np.array([self._low, self._low, self._low])
        self._high_array = np.array([self._high, self._high, self._high])

    def prepare_work(self, from_image: np.ndarray, dest_type=torch.float16) -> torch.Tensor:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = from_image
        low_mask = np.sum(image < self._low, axis=2) == 3
        high_mask = np.sum(image > self._high, axis=2) == 3

        image[low_mask] = self._low_array
        image[high_mask] = self._high_array
        image, _ = super().prepare_work(image, dest_type)
        image = image * 2.0 - 1.0

        return image

    def restore_work(self, from_image: torch.Tensor) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        image = (from_image / 2 + 0.5).clamp(0, 1)
        end_numpy, _ = super().restore_work(image)

        return end_numpy


class WorkerCompressorInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих сжатия."""

    @abstractmethod
    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        pass

    @abstractmethod
    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        pass

    @staticmethod
    def to_pillow_format(latent_img: torch.Tensor) -> Image.Image:
        """Преобразование из torch.Tensor в формат Pillow."""

        if latent_img.dtype in (torch.float16, torch.float32, torch.float64):
            latent_img *= 255
            latent_img = latent_img.to(dtype=torch.uint8)
        latent_img = latent_img.squeeze(0)
        this_shape = list(latent_img.shape)
        if this_shape[0] != 3:
            while this_shape[0] != 1:
                this_shape[0] //= 2
                if this_shape[1] < this_shape[2]:
                    this_shape[1] *= 2
                else:
                    this_shape[2] *= 2
            latent_img = latent_img.reshape(*this_shape)
        pillow_img: Image.Image = tvfunc.to_pil_image(latent_img)

        return pillow_img

    @staticmethod
    def from_pillow_format(pillow_img: Image.Image, dest_shape, dest_type) -> torch.Tensor:
        """Преобразование из формата Pillow в torch.Tensor."""

        this_size = pillow_img.height * pillow_img.width * 3
        real_size = reduce(lambda a, b: a * b, dest_shape)
        if (this_size // real_size) == 3:
            dest_mode = "L"
        else:
            dest_mode = "RGB"
        pillow_img = pillow_img.convert(mode=dest_mode)

        image = tvfunc.pil_to_tensor(pillow_img)
        if dest_type in (torch.float16, torch.float32, torch.float64):
            image = image.to(dtype=dest_type)
            image /= 255
            image.clamp_(0, 1)
        image = image.reshape(*dest_shape)

        return image

    @staticmethod
    def to_cv2_format(latent_img: torch.Tensor) -> numpy.ndarray:
        """Преобразование torch.Tensor в формат изображения cv2."""

        if latent_img.dtype in (torch.float16, torch.float32, torch.float64):
            latent_img *= 255
            latent_img = latent_img.to(dtype=torch.uint8)
        latent_img = latent_img.squeeze(0)
        this_shape = list(latent_img.shape)
        if this_shape[0] != 3:
            while this_shape[0] != 1:
                this_shape[0] //= 2
                if this_shape[1] < this_shape[2]:
                    this_shape[1] *= 2
                else:
                    this_shape[2] *= 2
            latent_img = latent_img.reshape(*this_shape)
        latent_img = latent_img.cpu()
        latent_img = latent_img.numpy()
        latent_img = np.moveaxis(latent_img, 0, 2)

        # Исправление крайне страшной проблемы
        # Иначе одинаковые тензоры с одинаковыми типами данных и устройствами
        # Дают разные результаты при сжатии и декомпрессии!!!
        crutch = np.copy(latent_img, order='C')
        return crutch

    @staticmethod
    def from_cv2_format(image: numpy.ndarray, dest_shape, dest_type) -> torch.Tensor:
        """Преобразование формата изображений cv2 в torch.Tensor."""

        if len(image.shape) == 3:
            image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image)
        if dest_type in (torch.float16, torch.float32, torch.float64):
            image = image.to(dtype=dest_type)
            image /= 255
            if dest_type == torch.float16:  # Иначе не работает на Linux
                image[image < 0.0] = 0
                image[image > 1.0] = 1
            else:
                image.clamp_(0, 1)

        image = image.reshape(*dest_shape)

        return image


class WorkerCompressorDummy(WorkerCompressorInterface):
    """Рабочий Deflated."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Deflated.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        return byter

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        latent_img = torch.frombuffer(compressed_bytes, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorDeflated(WorkerCompressorInterface):
    """Рабочий Deflated."""

    def __init__(self, level=9, device='cuda', *_, **__):
        self.level = level
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Deflated.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        obj = zlib.compressobj(level=self.level, method=zlib.DEFLATED)
        new_min = obj.compress(byter)
        new_min += obj.flush()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = zlib.decompress(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLzma(WorkerCompressorInterface):
    """Рабочий Lzma."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Lzma.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = lzma.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Lzma.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = lzma.decompress(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorGzip(WorkerCompressorInterface):
    """Рабочий Gzip."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Gzip.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = gzip.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Gzip.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = gzip.decompress(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorBzip2(WorkerCompressorInterface):
    """Рабочий Bzip2."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Bzip2.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = bz2.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Bzip2.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = bz2.decompress(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorZstd(WorkerCompressorInterface):
    """Рабочий ZStandard."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие ZStandard.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.zstd_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие ZStandard.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.zstd_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorBrotli(WorkerCompressorInterface):
    """Рабочий Brotli."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Brotli.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.brotli_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие Brotli.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.brotli_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLz4(WorkerCompressorInterface):
    """Рабочий LZ4."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZ4.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lz4_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZ4.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lz4_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLz4f(WorkerCompressorInterface):
    """Рабочий LZ4F."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZ4F.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lz4f_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZ4F.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lz4f_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLz4h5(WorkerCompressorInterface):
    """Рабочий LZ4H5."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZ4H5.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lz4h5_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZ4H5.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lz4h5_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLzw(WorkerCompressorInterface):
    """Рабочий LZW."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZW.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lzw_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZW.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lzw_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLzf(WorkerCompressorInterface):
    """Рабочий LZF."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZF.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lzf_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZF.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lzf_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorLzfse(WorkerCompressorInterface):
    """Рабочий LZFSE."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие LZFSE.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.lzfse_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие LZFSE.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.lzfse_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorAec(WorkerCompressorInterface):
    """Рабочий AEC."""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие AEC.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = latent_img.to('cpu')
        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = imagecodecs.aec_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие AEC.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        byters = imagecodecs.aec_decode(compressed_bytes)

        latent_img = torch.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)
        latent_img = latent_img.to(self.device)

        return latent_img


class WorkerCompressorH264(WorkerCompressorInterface):
    """Рабочий H264.
    Использовать только без автокодировщика и квантования!!!"""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие H264.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img *= 255.0
        image = latent_img.to(torch.uint8)
        _, channel, height, width = image.shape
        image = image.reshape(channel, height, width)
        image = image.cpu()
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        image = image.reshape(1, height, width, channel)

        buffer = io.BytesIO()
        writer = imageio.get_writer(buffer, format="mp4", codec="h264", fps=30)
        writer.append_data(image)
        writer.close()
        buffer.seek(0, 0)
        new_min = buffer.read()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие H264.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imageio.imread(compressed_bytes, format="pyav")

        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image)
        image = image.to(dest_type)
        image = image / 255.0
        image = image.reshape(*dest_shape)
        image = image.to(self.device)

        return image


class WorkerCompressorH265(WorkerCompressorInterface):
    """Рабочий H265.
    Использовать только без автокодировщика и квантования!!!"""

    def __init__(self, time_pad=0.1, device='cuda', *_, **__):
        self.device = device
        self.time_pad = time_pad
        # При попытке сборки мусора падают, так что сохраняем на время
        self._buggy_writers = []

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие H265.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img *= 255.0
        image = latent_img.to(torch.uint8)
        _, channel, height, width = image.shape
        image = image.reshape(channel, height, width)
        image = image.cpu()
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        image = image.reshape(1, height, width, channel)

        while True:
            buffer = io.BytesIO()
            buffer.seek(0, 0)
            writer = imageio.get_writer(buffer, format="mov", codec="hevc", fps=1)
            writer.append_data(image)
            try:  # Иногда падает
                writer.close()
                break
            except:
                print("=== !!! ОШИБКА С ЗАКРЫТИЕМ !!! ===")
                # writer.close()
                self._buggy_writers.append(writer)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие H265.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imageio.imread(compressed_bytes, format="pyav")

        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image)
        image = image.to(dest_type)
        image = image / 255.0
        image = image.reshape(*dest_shape)
        image = image.to(self.device)

        return image


class WorkerCompressorJpeg(WorkerCompressorInterface):
    """Рабочий JPEG."""

    def __init__(self, quality: Union[int, float, str] = 60, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие JPEG.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)

        buffer = io.BytesIO()
        imageio.imwrite(buffer, latent_img, format="JPEG-PIL", quality=self.quality)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие JPEG.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imageio.imread(compressed_bytes, format="JPEG-PIL")
        image = super().from_cv2_format(image, dest_shape, dest_type)
        image = image.to(device=self.device)

        return image


class WorkerCompressorAvif(WorkerCompressorInterface):
    """Рабочий AVIF."""

    def __init__(self, quality: Union[int, float, str] = 60, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие AVIF.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        pillow_img = super().to_pillow_format(latent_img)

        buffer = io.BytesIO()
        pillow_img.save(buffer, format="AVIF", optimize=True, quality=self.quality)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие AVIF.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        dummy_file = io.BytesIO(compressed_bytes)
        pillow_img = Image.open(dummy_file, formats=("AVIF",))
        dummy_file.close()

        image = super().from_pillow_format(pillow_img, dest_shape, dest_type)

        image = image.to(device=self.device)

        return image


class WorkerCompressorHeic(WorkerCompressorInterface):
    """Рабочий HEIC."""

    def __init__(self, quality: Union[int, float, str] = 60, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)

        pillow_heif.register_heif_opener()

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие HEIC.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        pillow_img = super().to_pillow_format(latent_img)

        buffer = io.BytesIO()
        pillow_img.save(buffer, format="HEIF", optimize=True, quality=self.quality)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие HEIC.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        dummy_file = io.BytesIO(compressed_bytes)
        pillow_img = Image.open(dummy_file, formats=("HEIF",))
        dummy_file.close()

        image = super().from_pillow_format(pillow_img, dest_shape, dest_type)

        image = image.to(device=self.device)

        return image


class WorkerCompressorWebp(WorkerCompressorInterface):
    """Рабочий WebP."""

    def __init__(self, lossless: Union[int, str, bool] = 1, quality: Union[int, float, str] = 60, device='cuda', *_, **__):
        self.device = device
        self.lossless = bool(int(lossless))
        self.quality = int(quality)

        pillow_heif.register_heif_opener()

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие WebP.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        pillow_img = super().to_pillow_format(latent_img)

        buffer = io.BytesIO()
        pillow_img.save(buffer, format="WebP", lossless=self.lossless, quality=self.quality)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие WebP.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        dummy_file = io.BytesIO(compressed_bytes)
        pillow_img = Image.open(dummy_file, formats=("WebP",))
        dummy_file.close()

        image = super().from_pillow_format(pillow_img, dest_shape, dest_type)

        image = image.to(device=self.device)

        return image


class WorkerCompressorJpegLS(WorkerCompressorInterface):
    """Рабочий JPEG LS."""

    def __init__(self, quality: Union[int, float, str] = 0, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие JPEG LS.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegls_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие JPEG LS.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imagecodecs.jpegls_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)
        image = image.to(device=self.device)

        return image


class WorkerCompressorJpegXR(WorkerCompressorInterface):
    """Рабочий JPEG XR."""

    def __init__(self, quality: Union[int, float, str] = 60, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие JPEG XR.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegxr_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие JPEG XR.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imagecodecs.jpegxr_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)
        image = image.to(device=self.device)

        return image


class WorkerCompressorJpegXL(WorkerCompressorInterface):
    """Рабочий JPEG XL."""

    def __init__(self, quality: Union[int, float, str] = 60, effort: Union[int, float, str] = 9, device='cuda', *_, **__):
        self.device = device
        self.quality = int(quality)
        self.effort = int(effort)

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие JPEG XL.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegxl_encode(latent_img, level=self.quality, effort=self.effort)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие JPEG XL.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imagecodecs.jpegxl_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)
        image = image.to(device=self.device)

        return image


class WorkerCompressorQoi(WorkerCompressorInterface):
    """Рабочий сжатия без потерь QOI.
    Только без автокодировщика!"""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие QOI.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)

        new_min = imagecodecs.qoi_encode(latent_img)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=torch.uint8) -> torch.Tensor:
        """Расжатие QOI.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде torch.Tensor."""

        image = imagecodecs.qoi_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        image = image.to(device=self.device)

        return image


class WorkerAutoencoderInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих-автокодировщиков."""

    z_shape = (1, 0, 0, 0)
    nominal_type = torch.float16

    @abstractmethod
    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        pass

    @abstractmethod
    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        pass


class WorkerAutoencoderVQ_F16(WorkerAutoencoderInterface):
    """Рабочий VAE VQ-f16."""

    z_shape = (1, 8, 32, 32)
    nominal_type = torch.float16

    def __init__(self, config_path: str, ckpt_path: str):
        """config_path -- путь к yaml-файлу конфигурации.
        ckpt_path -- путь к ckpt-файлу весов."""

        self._config_path = config_path
        self._ckpt_path = ckpt_path

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        self._model = instantiate_from_config(config.model)
        self._model.load_state_dict(sd, strict=False)
        self._model.eval()
        self._model = self._model.type(self.nominal_type).cuda()

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        latent, _, _ = self._model.encode(from_image)
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        to_image = self._model.decode(latent)
        return to_image


class WorkerAutoencoderVQ_F4(WorkerAutoencoderVQ_F16):
    """Рабочий VAE VQ-f4."""

    z_shape = (1, 3, 128, 128)
    nominal_type = torch.float16


class WorkerAutoencoderVQ_F8(WorkerAutoencoderVQ_F16):
    """Рабочий VAE VQ-f8."""

    z_shape = (1, 4, 64, 64)
    nominal_type = torch.float32


class WorkerAutoencoderVQ_F16_Optimized(WorkerAutoencoderInterface):
    """Рабочий оптимизированный VAE VQ-f16."""

    z_shape = (1, 8, 32, 32)
    nominal_type = torch.float16
    ts_base = "dependence/ts/"

    def __init__(self, config_path: str, ckpt_path: str,
                 decoder_name: str = "vq-f16_decoder_optim.ts",
                 encoder_name: str = "vq-f16_encoder_optim.ts"):
        """config_path -- путь к yaml-файлу конфигурации.
        ckpt_path -- путь к ckpt-файлу весов."""

        self._config_path = config_path
        self._ckpt_path = ckpt_path

        self._decoder_path = os.path.join(self.ts_base, decoder_name)
        self._encoder_path = os.path.join(self.ts_base, encoder_name)
        is_decoder = os.path.isfile(self._decoder_path)
        is_encoder = os.path.isfile(self._encoder_path)

        torch._C._jit_set_profiling_executor(False)
        if is_decoder:
            self._decoder_model = torch.jit.load(self._decoder_path).cuda()
        else:
            model = self._create_model(config_path, ckpt_path)
            model.forward = model.decode
            model._trainer = pl.Trainer()
            inp = [torch.randn(1, 8, 32, 32, dtype=self.nominal_type, device='cuda')]
            traced_model = torch.jit.trace(model, inp)
            torch.jit.save(traced_model, self._decoder_path)
            self._decoder_model = traced_model.cuda()
        self._decoder_model.eval()

        if is_encoder:
            self._encoder_model = torch.jit.load(self._encoder_path).cuda()
        else:
            model = self._create_model(config_path, ckpt_path)
            model.forward = model.encode
            model._trainer = pl.Trainer()
            inp = [torch.randn(1, 3, 512, 512, dtype=self.nominal_type, device='cuda')]
            traced_model = torch.jit.trace(model, inp)
            torch.jit.save(traced_model, self._encoder_path)
            self._encoder_model = traced_model.cuda()
        self._encoder_model.eval()

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        latent, _, _ = self._encoder_model.forward(from_image)
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        to_image = self._decoder_model.forward(latent)
        return to_image

    # Технический метод

    def _create_model(self, config: str, ckpt: str):
        """Создание оптимизированной модели.
        config -- файл конфигурации.
        ckpt -- файл весов."""

        config = OmegaConf.load(f"{config}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.eval()
        model = model.to(self.nominal_type).cuda()

        return model


class WorkerAutoencoderKL_F16(WorkerAutoencoderInterface):
    """Рабочий VAE KL-f16."""

    z_shape = (1, 16, 32, 32)
    nominal_type = torch.float16

    def __init__(self, config_path: str, ckpt_path: str):
        """config_path -- путь к yaml-файлу конфигурации.
        ckpt_path -- путь к ckpt-файлу весов."""

        self._config_path = config_path
        self._ckpt_path = ckpt_path

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        self._model = instantiate_from_config(config.model)
        self._model.load_state_dict(sd, strict=False)
        self._model.eval()
        self._model = self._model.type(self.nominal_type).cuda()

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        gauss = self._model.encode(from_image)
        latent = gauss.sample().type(self.nominal_type)
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        to_image = self._model.decode(latent)
        return to_image


class WorkerAutoencoderKL_F4(WorkerAutoencoderKL_F16):
    """Рабочий VAE KL-f4."""

    z_shape = (1, 3, 128, 128)
    nominal_type = torch.float16


class WorkerAutoencoderKL_F8(WorkerAutoencoderKL_F16):
    """Рабочий VAE KL-f8."""

    z_shape = (1, 4, 64, 64)
    nominal_type = torch.float16


class WorkerAutoencoderKL_F32(WorkerAutoencoderKL_F16):
    """Рабочий VAE KL-f32."""

    z_shape = (1, 64, 16, 16)
    nominal_type = torch.float16


class WorkerAutoencoderCDC(WorkerAutoencoderInterface):
    """Рабочий CDC."""

    z_shape = (1, 256, 32, 32)
    nominal_type = torch.float16

    def __init__(self, config_path: str, ckpt_path: str):
        """config_path -- путь к yaml-файлу конфигурации (не существует, Для совместимости с интерфейсом).
        ckpt_path -- путь к ckpt-файлу весов."""

        self._ckpt_path = ckpt_path

        self._lpips_weight = 0.0
        self._denoise_step = 20
        self._gamma = 0.8

        self.denoise_model = Unet(
            dim=64,
            channels=3,
            context_channels=3,
            dim_mults=(1, 2, 3, 4, 5, 6),
            context_dim_mults=(1, 2, 3, 4),
        )

        self.compressor = BigCompressor(
            dim=64,
            dim_mults=(1, 2, 3, 4),
            hyper_dims_mults=(4, 4, 4),
            channels=3,
            out_channels=3,
            vbr=False,
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_model,
            context_fn=self.compressor,
            num_timesteps=20000,
            loss_type="l1",
            clip_noise="none",
            vbr=False,
            lagrangian=0.9,
            pred_mode="noise",
            var_schedule="linear",
            aux_loss_weight=self._lpips_weight,
            aux_loss_type="lpips"
        )

        loaded_param = torch.load(
            self._ckpt_path,
            map_location=lambda storage, loc: storage,
        )
        self.diffusion.load_state_dict(loaded_param["model"])
        self.diffusion.eval()
        self.diffusion = self.diffusion.float().cuda()
        self.batcher = 1

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        self.batcher = from_image.shape[0]
        from_image = from_image.float()
        latent = self.compressor.encode(from_image)[0]
        latent = latent.half()
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        latent = latent.float()
        image = self.compressor.decode(latent)

        self.diffusion.set_sample_schedule(
            self._denoise_step,
            "cuda",
        )
        initer = torch.randn(self.batcher, 3, 512, 512) * self._gamma
        initer = initer.cuda().float()
        to_image = self.diffusion.p_sample_loop(
            (self.batcher, 3, 512, 512), image, sample_mode="ddim", init=initer, eta=0
        )
        to_image = to_image.clamp(-1, 1)

        to_image = to_image.half()
        return to_image


class WorkerQuantInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих квантования."""

    @abstractmethod
    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        pass

    @abstractmethod
    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        pass

    def pre_quantize(self, new_img: torch.Tensor):
        """Выполнить предварительное квантование float32 -> float32."""

        pre_device = new_img.device
        pre_dtype = new_img.dtype
        new_img = new_img.to(device="cpu", dtype=torch.float32)
        new_img = new_img.numpy()
        new_img = imagecodecs.quantize_encode(new_img, mode=self.pre_quant, nsd=self.nsd)
        new_img = torch.from_numpy(new_img).to(device=pre_device, dtype=pre_dtype)

        return new_img


class WorkerQuantLinear(WorkerQuantInterface):
    """Класс для линейного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: Union[int, str] = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 47.69)
            # self.quant_params = (-25.53125, 4.614079728583546)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            maxer = new_img.max().item()
            miner = new_img.min().item()
            aller = maxer - miner
            scaler = 255 / aller

        new_img = (new_img - miner) * scaler
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = (new_img / scaler) + miner

        return new_img


class WorkerQuantPower(WorkerQuantInterface):
    """Класс для степенного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 3.31)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            maxer = new_img.max().item()
            miner = new_img.min().item()
            aller = maxer - miner
            scaler = math.log(255, aller)

        new_img = (new_img - miner) ** scaler
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = (new_img ** (1 / scaler)) + miner

        return new_img


class WorkerQuantLogistics(WorkerQuantInterface):
    """Класс для логистического квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (0.055, 256.585)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner = self.quant_params[0]
        else:
            meaner = new_img.mean().item()

        new_img -= meaner
        new_img = 1 / (1 + torch.exp(-new_img))
        new_max = torch.max(new_img).item()

        if self._hardcore:
            scaler = self.quant_params[1]
        else:
            scaler = 255 / new_max
        new_img *= scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img /= scaler
        new_img = -torch.log((1 / new_img) - 1)
        new_img += miner

        return new_img


class WorkerQuantMinLogistics(WorkerQuantInterface):
    """Класс для модифицированного (min вместо mean) логистического квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 256.585)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner = self.quant_params[0]
        else:
            miner = new_img.min().item()

        new_img -= miner
        new_img = 1 / (1 + torch.exp(-new_img))
        new_max = torch.max(new_img).item()

        if self._hardcore:
            scaler = self.quant_params[1]
        else:
            scaler = 255 / new_max
        new_img *= scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img /= scaler
        new_img = -torch.log((1 / new_img) - 1)
        new_img += miner

        return new_img


class WorkerQuantOddPower(WorkerQuantInterface):
    """Класс для нечётностепенного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, power: Union[int, float, str] = 3, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.power = int(power)
        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            scale_param = 255 / (2 * 2.5**self.power)
            self.quant_params = (0.055, scale_param)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / (2*new_img.min().item()**self.power)

        new_img -= meaner
        new_img = scaler*(new_img**self.power) + 127.5
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = ((new_img - 127.5) / scaler)
        mask = new_img < 0
        new_img = torch.abs(new_img) ** (1/self.power)
        new_img[mask] *= -1
        new_img += meaner

        return new_img


class WorkerQuantTanh(WorkerQuantInterface):
    """Класс для тангенсуального квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (0.055, 255/2)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / 2

        new_img -= meaner
        new_img = (torch.tanh(new_img) + 1)*scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = torch.arctanh(new_img / scaler - 1)
        new_img += meaner

        return new_img


class WorkerQuantMinTanh(WorkerQuantInterface):
    """Класс для модифицированного (mean -> min) тангенсуального квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 253/2)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner = new_img.min().item()
            scaler = 255 / 2

        new_img -= miner
        new_img = (torch.tanh(new_img) + 1)*scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = new_img / scaler - 1
        new_img[new_img == 1.0] = 0.999
        new_img = torch.arctanh(new_img)
        new_img += meaner

        return new_img


class WorkerQuantDoubleLogistics(WorkerQuantInterface):
    """Класс для двойного логистического квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (0.055, 255/2)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / 2

        new_img -= meaner
        new_img = (torch.sign(new_img)*(1 - torch.exp(-(new_img**2))) + 1) * scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        # new_img = -torch.log(1 - ((new_img / scaler - 1) / torch.sign(new_img)))
        new_img = torch.abs(1 - (new_img / scaler - 1))
        new_img[new_img == 0] = 0.005
        new_img = -torch.log(new_img)
        mask = new_img < 0
        new_img = torch.abs(new_img)
        new_img = torch.sqrt(new_img)
        new_img[mask] *= -1
        new_img += meaner

        return new_img


class WorkerQuantMinDoubleLogistics(WorkerQuantInterface):
    """Класс для модифицированного (mean -> min) двойного логистического квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 255/2)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner = new_img.min().item()
            scaler = 255 / 2

        new_img -= miner
        new_img = (torch.sign(new_img)*(1 - torch.exp(-(new_img**2))) + 1) * scaler
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        # new_img = -torch.log(1 - ((new_img / scaler - 1) / torch.sign(new_img)))
        new_img = torch.abs(1 - (new_img / scaler - 1))
        new_img[new_img == 0] = 0.005
        new_img = -torch.log(new_img)
        mask = new_img < 0
        new_img = torch.abs(new_img)
        new_img = torch.sqrt(new_img)
        new_img[mask] *= -1
        new_img += meaner

        return new_img


class WorkerQuantSinh(WorkerQuantInterface):
    """Класс для гиперболическосинусоидального квантования и деквантования с нормализированными параметрами."""

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.quant_params = (0.055, 21.0737)
        self._hardcore = hardcore

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        new_img = torch.clone(latent)
        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / (2*torch.sinh(new_img.min()).item())

        new_img -= meaner
        new_img = scaler*torch.sinh(new_img) + 127.5
        new_img = torch.round(new_img)
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, dest_type=torch.float16, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(dest_type)
        new_img = torch.arcsinh((new_img - 127.5) / scaler)
        new_img += meaner

        return new_img


class WorkerSRInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих суперрезолюции."""

    @abstractmethod
    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        pass


class WorkerSRDummy(WorkerSRInterface):
    """Ложный класс работника суперрезолюции."""

    def __init__(self, config_path: str = "", ckpt_path: str = "", dest_height: int = 720, dest_width: int = 1280):
        """dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        self._dest_size = (dest_width, dest_height)

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if not dest_size:
            dest_size = self._dest_size
        if img.shape[::-1][1:] != tuple(dest_size):
            new_img = cv2.resize(img, dest_size, interpolation=cv2.INTER_CUBIC)
        else:
            new_img = np.copy(img)

        return new_img


class WorkerSRRealESRGAN_x2plus(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 2

    def __init__(self, config_path: str, ckpt_path: str,
                 dni_base: float = 0.75, dest_height: int = 720, dest_width: int = 1280):
        """path -- путь к pth-файлу весов модели.
        dni_base -- основной уровень шума (0-1).
        dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""
        self._backend_model = backend_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                                      num_block=23, num_grow_ch=32, scale=self.this_scale)

        dni_base = dni_base
        dni_weight = [dni_base, 1 - dni_base]
        self._model = RealESRGANer(scale=self.this_scale, model_path=ckpt_path, dni_weight=dni_weight, tile=0, tile_pad=10,
                                   pre_pad=0,
                                   model=backend_model, half=False)
        self._dest_size = [dest_width // self.this_scale, dest_height // self.this_scale]

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if dest_size:
            dest_size = list(map(lambda x: x // self.this_scale, dest_size))
        else:
            dest_size = self._dest_size
        new_img = cv2.resize(img, dest_size)
        new_img = self._model.enhance(new_img, outscale=2)[0]

        return new_img


class WorkerSRAPISR_RRDB_x2(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 2
    nominal_type = torch.float16

    def __init__(self, config_path: str, ckpt_path: str, dest_height: int = 720, dest_width: int = 1280):
        """path -- путь к pth-файлу весов модели.
        dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        self._dest_size = [dest_width // self.this_scale, dest_height // self.this_scale]
        self._model = load_rrdb(ckpt_path, scale=self.this_scale)
        self._model = self._model.cuda().to(dtype=self.nominal_type)
        self._to_tensor = torchvision.transforms.ToTensor()

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if dest_size:
            dest_size = list(map(lambda x: x // self.this_scale, dest_size))
        else:
            dest_size = self._dest_size
        new_img = cv2.resize(img, dest_size)
        new_img = self._to_tensor(new_img).unsqueeze(0).cuda()
        new_img = new_img.to(dtype=self.nominal_type)
        new_img = self._model(new_img)
        new_img *= 255.0
        new_img = new_img.to(torch.uint8)
        new_img = new_img.squeeze(0)
        new_img = new_img.cpu()
        new_img = new_img.numpy()
        new_img = np.moveaxis(new_img, 0, 2)

        return new_img


class WorkerSRAPISR_RRDB_x2_Optimized(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 2
    nominal_type = torch.float16
    ts_base = "dependence/ts/"

    def __init__(self, config_path: str, ckpt_path: str, dest_height: int = 720, dest_width: int = 1280,
                 sr_name: str = "apisr_rrdb_x2_optim.ts"):
        """path -- путь к pth-файлу весов модели.
        dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        self._dest_size = [dest_width // self.this_scale, dest_height // self.this_scale]
        self._sr_path = os.path.join(self.ts_base, sr_name)
        self._to_tensor = torchvision.transforms.ToTensor()

        is_sr = os.path.isfile(self._sr_path)

        torch._C._jit_set_profiling_executor(False)
        if is_sr:
            self._model = torch.jit.load(self._sr_path).cuda()
        else:
            model = load_rrdb(ckpt_path, scale=self.this_scale).half()
            inp = [torch.randn(1, 3, 512, 512, dtype=torch.float16, device='cuda')]
            traced_model = torch.jit.trace(model, inp)
            torch.jit.save(traced_model, self._sr_path)
            self._model = traced_model.cuda()
        self._model.eval()

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if dest_size:
            dest_size = list(map(lambda x: x // self.this_scale, dest_size))
        else:
            dest_size = self._dest_size
        new_img = cv2.resize(img, dest_size)
        new_img = self._to_tensor(new_img).unsqueeze(0).cuda()
        new_img = new_img.to(dtype=self.nominal_type)
        new_img = self._model(new_img)
        new_img *= 255.0
        new_img = new_img.to(torch.uint8)
        new_img = new_img.squeeze(0)
        new_img = new_img.cpu()
        new_img = new_img.numpy()
        new_img = np.moveaxis(new_img, 0, 2)

        return new_img


class WorkerSRAPISR_GRL_x4(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 4
    nominal_type = torch.float32

    def __init__(self, config_path: str, ckpt_path: str, dest_height: int = 720, dest_width: int = 1280):
        """path -- путь к pth-файлу весов модели.
        dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        # self._dest_size = [dest_width // self.this_scale, dest_height // self.this_scale]
        self._dest_width = dest_width
        self._dest_height = dest_height
        self._dest_size = [640, 360]
        self._model = load_grl(ckpt_path, scale=self.this_scale)
        self._model = self._model.cuda().to(dtype=self.nominal_type)
        self._to_tensor = torchvision.transforms.ToTensor()

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if dest_size:
            dest_size = list(map(lambda x: x // self.this_scale, dest_size))
        else:
            dest_size = self._dest_size
        new_img = cv2.resize(img, dest_size)
        new_img = self._to_tensor(new_img).unsqueeze(0).cuda()
        new_img = new_img.to(dtype=self.nominal_type)
        new_img = self._model(new_img)
        new_img *= 255.0
        new_img = new_img.to(torch.uint8)
        new_img = new_img.squeeze(0)
        new_img = new_img.cpu()
        new_img = new_img.numpy()
        new_img = np.moveaxis(new_img, 0, 2)
        new_img = cv2.resize(new_img, [self._dest_width, self._dest_height])

        return new_img


class WorkerPredictorInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих-предикторов."""

    @abstractmethod
    def predict_work(self, images: list[np.ndarray], predict_num: int = 1) -> list[np.ndarray]:
        """Вход: список картинок в формате cv2 (np.ndarray), число картинок для предсказания.
        Выход: список предсказанных картинок."""

        pass


class WorkerPredictorDummy(WorkerPredictorInterface):
    """Класс ложного ("ленивого") рабочего предиктора."""

    def __init__(self, *_, **__):
        pass

    def predict_work(self, images: list[np.ndarray], predict_num: int = 1) -> list[np.ndarray]:
        """Вход: список картинок в формате cv2 (np.ndarray), число картинок для предсказания.
        Выход: список предсказанных картинок."""

        predict_images = []
        if len(images) < predict_num:  # Для небольшой оптимизации
            predict_images = copy.deepcopy(images[:predict_num])
        else:
            while len(predict_images) < predict_num:
                predict_images += copy.deepcopy(images)
            predict_images = predict_images[:predict_num]

        return predict_images


class WorkerPredictorDMVFN(WorkerPredictorInterface):
    """Класс рабочего предиктора на основе модели DMVFN."""

    def __init__(self, path: str):
        """path -- путь к pth-файлу весов модели."""

        self._backend_model = DMVFN(load_path=path).cuda()
        self._model = Predictor(self._backend_model)

    def predict_work(self, images: list[np.ndarray], predict_num: int = 1) -> list[np.ndarray]:
        """Вход: список картинок в формате cv2 (np.ndarray), число картинок для предсказания.
        Выход: список предсказанных картинок."""

        if len(images) == 1:
            images *= (predict_num + 1)
        predict_images = self._model.predict(images, predict_num)

        if not isinstance(predict_images, list):
            predict_images = [predict_images]

        return predict_images


if __name__ == "__main__":
    test = WorkerDummy()
    result = test.do_work()
    print(result)
