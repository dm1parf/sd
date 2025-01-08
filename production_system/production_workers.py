import copy
import math
import time
import zlib
import lzma
import bz2
import gzip
import io
from typing import Union
from abc import abstractmethod
import cv2
import imagecodecs
import pillow_heif
from PIL import Image
import numpy as np
import onnxruntime


# WorkerDummy -- класс ложного ("ленивого") рабочего, имитирующего деятельность

# > WorkerASInterface -- абстрактный класс подавителя артефактов
# WorkerASDummy -- класс ленивого подавителя артефактов, просто делает переводы форматов и ничего не подавляет
# WorkerASCutEdgeColors -- класс подавителя артефактов, что обрезает цвета
# WorkerASMoveDistribution -- класс подавителя артефактов, что переносит распределение
# WorkerASComposit -- класс подавителя артефактов, что переносит распределение

# > WorkerAutoencoderInterface -- абстрактный класс интерфейса для автокодировщиков
# WorkerAutoencoderKL_F4 -- класс рабочего вариационного автокодировщика KL-f4
# WorkerAutoencoderKL_F16 -- класс рабочего вариационного автокодировщика KL-f16

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

# > WorkerPredictorInterface -- абстрактный класс интерфейса для предиктора
# WorkerPredictorDummy -- класс ложного ("ленивого") рабочего, имитирующего предиктор


class WorkerDummy():
    """Ложный рабочий."""

    def __init__(self, wait=0.5, verbose=True):
        self.wait = wait
        self._verbose = verbose

    def do_work(self, *_, **__):
        time.sleep(self.wait)
        if self._verbose:
            print("...")


class WorkerASInterface:
    """Интерфейс для рабочих-подавителей артефактов."""

    @abstractmethod
    def prepare_work(self, from_image: np.ndarray, dest_type=np.float16) -> np.ndarray:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = cv2.cvtColor(from_image, cv2.COLOR_BGR2RGB)
        if image.shape[::-1][1:] != (self.middle_width, self.middle_height):
            image = cv2.resize(image, (self.middle_width, self.middle_height), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, 2, 0)

        image = image.astype(dest_type)
        current_shape = image.shape
        image = image / 255.0
        image = image.reshape(1, *current_shape)

        return image

    @abstractmethod
    def restore_work(self, from_image: np.ndarray) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        from_image *= 255.0
        image = from_image.astype(np.uint8)
        
        image = image.reshape(3, self.middle_height, self.middle_width)

        image = np.moveaxis(image, 0, 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


class WorkerASDummy(WorkerASInterface):
    """Интерфейс для рабочих-подавителей артефактов."""

    def __init__(self, middle_width: int = 512, middle_height: int = 512):
        """
        :param middle_width: Промежуточная ширина.
        :param middle_height: Промежуточная высота.
        """

        self.middle_width = int(middle_width)
        self.middle_height = int(middle_height)

    def prepare_work(self, from_image: np.ndarray, dest_type=np.float16) -> np.ndarray:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = super().prepare_work(from_image, dest_type)

        return image

    def restore_work(self, from_image: np.ndarray) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        end_numpy = super().restore_work(from_image)

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

    def prepare_work(self, from_image: np.ndarray, dest_type=np.float16) -> np.ndarray:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = from_image
        low_mask = np.sum(image < self._low, axis=2) == 3
        high_mask = np.sum(image > self._high, axis=2) == 3

        image[low_mask] = self._low_array
        image[high_mask] = self._high_array

        image = super().prepare_work(image, dest_type)

        return image

    def restore_work(self, from_image: np.ndarray) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        end_numpy = super().restore_work(from_image)

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

    def prepare_work(self, from_image: np.ndarray, dest_type=np.float16) -> np.ndarray:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = super().prepare_work(from_image, dest_type)
        image = image * 2.0 - 1.0

        return image

    def restore_work(self, from_image: np.ndarray) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = (from_image / 2 + 0.5).clip(0, 1)
        end_numpy = super().restore_work(image)

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

    def prepare_work(self, from_image: np.ndarray, dest_type=np.float16) -> np.ndarray:
        """Подготовка картинки для подавления артефактов далее.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = from_image
        low_mask = np.sum(image < self._low, axis=2) == 3
        high_mask = np.sum(image > self._high, axis=2) == 3

        image[low_mask] = self._low_array
        image[high_mask] = self._high_array
        image = super().prepare_work(image, dest_type)
        image = image * 2.0 - 1.0

        return image

    def restore_work(self, from_image: np.ndarray) -> np.ndarray:
        """Восстановление картинки после подавления артефактов.
        Вход: картинка в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        image = (from_image / 2 + 0.5).clip(0, 1)
        end_numpy = super().restore_work(image)

        return end_numpy


class WorkerCompressorInterface():
    """Интерфейс для рабочих сжатия."""

    @abstractmethod
    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        pass

    @abstractmethod
    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        pass

    def to_pillow_format(self, latent_img: np.ndarray) -> Image.Image:
        """Преобразование из np.ndarray в формат Pillow."""

        cv2_img = self.to_cv2_format(latent_img)
        pillow_img: Image.Image = Image.fromarray(cv2_img)

        return pillow_img

    def from_pillow_format(self, pillow_img: Image.Image, dest_shape, dest_type) -> np.ndarray:
        """Преобразование из формата Pillow в np.ndarray."""

        cv2_img = np.array(pillow_img)
        image = self.from_cv2_format(cv2_img, dest_shape, dest_type)

        return image

    @staticmethod
    def to_cv2_format(latent_img: np.ndarray) -> np.ndarray:
        """Преобразование np.ndarray в формат изображения cv2."""

        if latent_img.dtype in (np.float16, np.float32, np.float64):
            latent_img *= 255
            latent_img = latent_img.astype(dtype=np.uint8)
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
        latent_img = np.moveaxis(latent_img, 0, 2)

        # Исправление крайне страшной проблемы
        # Иначе одинаковые тензоры с одинаковыми типами данных и устройствами
        # Дают разные результаты при сжатии и декомпрессии!!!
        # crutch = np.copy(latent_img, order='C')
        # return crutch

        return latent_img

    @staticmethod
    def from_cv2_format(image: np.ndarray, dest_shape, dest_type) -> np.ndarray:
        """Преобразование формата изображений cv2 в np.ndarray."""

        if len(image.shape) == 3:
            image = np.moveaxis(image, 2, 0)
        if dest_type in (np.float16, np.float32, np.float64):
            image = image.astype(dtype=dest_type)
            image /= 255
            if dest_type == np.float16:  # Иначе не работает на Linux
                image[image < 0.0] = 0
                image[image > 1.0] = 1
            else:
                image = image.clip(0, 1)

        image = image.reshape(*dest_shape)

        return image


class WorkerCompressorDummy(WorkerCompressorInterface):
    """Рабочий Deflated."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Deflated.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')

        return byter

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        latent_img = np.frombuffer(compressed_bytes, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorDeflated(WorkerCompressorInterface):
    """Рабочий Deflated."""

    def __init__(self, level=9, *_, **__):
        self.level = level

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Deflated.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')

        obj = zlib.compressobj(level=self.level, method=zlib.DEFLATED)
        new_min = obj.compress(byter)
        new_min += obj.flush()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Deflated.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = zlib.decompress(compressed_bytes)

        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLzma(WorkerCompressorInterface):
    """Рабочий Lzma."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Lzma.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = lzma.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Lzma.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = lzma.decompress(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorGzip(WorkerCompressorInterface):
    """Рабочий Gzip."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Gzip.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = gzip.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Gzip.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = gzip.decompress(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorBzip2(WorkerCompressorInterface):
    """Рабочий Bzip2."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Bzip2.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = bz2.compress(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Bzip2.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = bz2.decompress(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorZstd(WorkerCompressorInterface):
    """Рабочий ZStandard."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие ZStandard.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.zstd_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие ZStandard.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.zstd_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorBrotli(WorkerCompressorInterface):
    """Рабочий Brotli."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие Brotli.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.brotli_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие Brotli.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.brotli_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLz4(WorkerCompressorInterface):
    """Рабочий LZ4."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZ4.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lz4_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZ4.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lz4_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLz4f(WorkerCompressorInterface):
    """Рабочий LZ4F."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZ4F.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lz4f_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZ4F.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lz4f_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLz4h5(WorkerCompressorInterface):
    """Рабочий LZ4H5."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZ4H5.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lz4h5_encode(byter, level=9)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZ4H5.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lz4h5_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLzw(WorkerCompressorInterface):
    """Рабочий LZW."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZW.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lzw_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZW.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lzw_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLzf(WorkerCompressorInterface):
    """Рабочий LZF."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZF.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lzf_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZF.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lzf_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorLzfse(WorkerCompressorInterface):
    """Рабочий LZFSE."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие LZFSE.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.lzfse_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие LZFSE.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.lzfse_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorAec(WorkerCompressorInterface):
    """Рабочий AEC."""

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие AEC.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        byter = latent_img.tobytes(order='C')
        new_min = imagecodecs.aec_encode(byter)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие AEC.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        byters = imagecodecs.aec_decode(compressed_bytes)
        latent_img = np.frombuffer(byters, dtype=dest_type)
        latent_img = latent_img.reshape(dest_shape)

        return latent_img


class WorkerCompressorJpeg(WorkerCompressorInterface):
    """Рабочий JPEG."""

    def __init__(self, quality: Union[int, float, str] = 60, *_, **__):
        self.quality = int(quality)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие JPEG.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpeg_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие JPEG.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.jpeg_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorAvif(WorkerCompressorInterface):
    """Рабочий AVIF."""

    def __init__(self, quality: Union[int, float, str] = 60, *_, **__):
        self.quality = int(quality)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие AVIF.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.avif_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие AVIF.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.avif_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorHeic(WorkerCompressorInterface):
    """Рабочий HEIC."""

    def __init__(self, quality: Union[int, float, str] = 60, *_, **__):
        self.quality = int(quality)

        pillow_heif.register_heif_opener()

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие HEIC.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        pillow_img = super().to_pillow_format(latent_img)

        buffer = io.BytesIO()
        pillow_img.save(buffer, format="HEIF", optimize=True, quality=self.quality)
        buffer.seek(0, 0)
        new_min = buffer.read()
        buffer.close()

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие HEIC.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        dummy_file = io.BytesIO(compressed_bytes)
        pillow_img = Image.open(dummy_file, formats=("HEIF",))
        dummy_file.close()
        image = super().from_pillow_format(pillow_img, dest_shape, dest_type)

        return image


class WorkerCompressorWebp(WorkerCompressorInterface):
    """Рабочий WebP."""

    def __init__(self, lossless: Union[int, str, bool] = 1, quality: Union[int, float, str] = 60, *_, **__):
        self.lossless = bool(int(lossless))
        self.quality = int(quality)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие WebP.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.webp_encode(latent_img, level=self.quality, lossless=self.lossless)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие WebP.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.webp_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorJpegLS(WorkerCompressorInterface):
    """Рабочий JPEG LS."""

    def __init__(self, quality: Union[int, float, str] = 0, *_, **__):
        self.quality = int(quality)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие JPEG LS.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegls_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие JPEG LS.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.jpegls_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorJpegXR(WorkerCompressorInterface):
    """Рабочий JPEG XR."""

    def __init__(self, quality: Union[int, float, str] = 60, *_, **__):
        self.quality = int(quality)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие JPEG XR.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegxr_encode(latent_img, level=self.quality)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие JPEG XR.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.jpegxr_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorJpegXL(WorkerCompressorInterface):
    """Рабочий JPEG XL."""

    def __init__(self, quality: Union[int, float, str] = 60, effort: Union[int, float, str] = 9, *_, **__):
        self.quality = int(quality)
        self.effort = int(effort)

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие JPEG XL.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)
        new_min = imagecodecs.jpegxl_encode(latent_img, level=self.quality, effort=self.effort)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие JPEG XL.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.jpegxl_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerCompressorQoi(WorkerCompressorInterface):
    """Рабочий сжатия без потерь QOI.
    Только без автокодировщика!"""

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: np.ndarray) -> bytes:
        """Сжатие QOI.
        Вход: картинка в виде np.ndarray.
        Выход: bytes."""

        latent_img = super().to_cv2_format(latent_img)

        new_min = imagecodecs.qoi_encode(latent_img)

        return new_min

    def decompress_work(self, compressed_bytes: bytes, dest_shape: tuple, dest_type=np.uint8) -> np.ndarray:
        """Расжатие QOI.
        Вход: bytes, итоговая форма, итоговый тип данных.
        Выход: картинка в виде np.ndarray."""

        image = imagecodecs.qoi_decode(compressed_bytes)
        image = super().from_cv2_format(image, dest_shape, dest_type)

        return image


class WorkerAutoencoderInterface:
    """Интерфейс для рабочих-автокодировщиков."""

    z_shape = (1, 0, 0, 0)
    nominal_type = np.float16
    nominal_name = "AutoencoderInterface"
    timing_cache_path = "./dependence/onnx/timing_cache"
    engine_cache_path = "./dependence/onnx/engine_cache"

    def __init__(self, encoder_path: str, decoder_path: str, max_worksize: int):
        """coder_path -- путь к ONNX-модели кодера.
        decoder_path -- путь к ONNX-модели декодера.
        max_worksize -- объём видеопамяти."""

        self.providers = [
            # 37_580_963_840  # 35 Gb
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': max_worksize,
                'trt_fp16_enable': True,

                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": self.timing_cache_path,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": self.engine_cache_path,
            }),
        ]

        # Если "" или none, то не загружается.
        if encoder_path:
            self._encoder_session = onnxruntime.InferenceSession(encoder_path, providers=self.providers)
        else:
            self._encoder_session = None
        if decoder_path:
            self._decoder_session = onnxruntime.InferenceSession(decoder_path, providers=self.providers)
        else:
            self._decoder_session = None

    # @abstractmethod
    def encode_work(self, from_image: np.ndarray) -> np.ndarray:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде np.ndarray.
        Выход: латентное пространство в виде np.ndarray."""

        if not self._encoder_session:
            raise ModuleNotFoundError("Encoder was not loaded!")
        latent = self._encoder_session.run(None, {"input": from_image})[0]

        return latent

    # @abstractmethod
    def decode_work(self, latent: np.ndarray) -> np.ndarray:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде np.ndarray.
        Выход: картинка в виде np.ndarray."""

        if not self._decoder_session:
            raise ModuleNotFoundError("Decoder was not loaded!")
        from_image = self._decoder_session.run(None, {"input": latent})[0]

        return from_image


class WorkerAutoencoderKL_F16(WorkerAutoencoderInterface):
    """Рабочий VAE KL-f16."""

    z_shape = (1, 16, 32, 32)
    nominal_type = np.float16
    nominal_name = "AutoencoderKL_F16"


class WorkerAutoencoderKL_F4(WorkerAutoencoderInterface):
    """Рабочий VAE KL-f4."""

    z_shape = (1, 3, 128, 128)
    nominal_type = np.float16
    nominal_name = "AutoencoderKL_F4"


class WorkerQuantInterface:
    """Интерфейс для рабочих квантования."""

    quant_params_dict = {
        "default": tuple(),
    }

    @abstractmethod
    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        pass

    @abstractmethod
    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        pass

    def pre_quantize(self, new_img: np.ndarray):
        """Выполнить предварительное квантование float32 -> float32."""

        pre_dtype = new_img.dtype
        new_img = new_img.astype(np.float32)
        new_img = imagecodecs.quantize_encode(new_img, mode=self.pre_quant, nsd=self.nsd)
        new_img = new_img.astype(pre_dtype)

        return new_img

    def unlock(self) -> None:
        """Разблокировать параметры."""

        self._hardcore = False

    def adjust_params(self, autoencoder_worker: str = "default") -> tuple:
        """Настроить параметры под конкретный вариационный автокодировщик."""

        new_params = self.quant_params_dict.get(autoencoder_worker, None)
        if not new_params:
            new_params = self.quant_params_dict["default"]
        self.quant_params = new_params
        self._hardcore = True

        return self.quant_params


class WorkerQuantLinear(WorkerQuantInterface):
    """Класс для линейного квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (-2.41, 47.69),
        "AutoencoderVQ_F4": (-4.0751, 35.5137),
        "AutoencoderVQ_F8": (-2.6578, 41.7294),
        "AutoencoderVQ_F16": (-2.5526, 47.5477),
        "AutoencoderKL_F4": (-67.7418, 1.9253),
        "AutoencoderKL_F8": (-31.1259, 4.2108),
        "AutoencoderKL_F16": (-20.9596, 7.7829),
        "AutoencoderKL_F32": (-6.0891, 21.8568),
    }

    def __init__(self, pre_quant: str = "", nsd: Union[int, str] = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

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
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = (new_img / scaler) + miner

        return new_img


class WorkerQuantPower(WorkerQuantInterface):
    """Класс для степенного квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (-2.41, 3.31),
        "AutoencoderVQ_F4": (-4.0751, 2.8109),
        "AutoencoderVQ_F8": (-2.6578, 3.0614),
        "AutoencoderVQ_F16": (-2.5526, 3.2997),
        "AutoencoderKL_F4": (-67.7418, 1.1333),
        "AutoencoderKL_F8": (-31.1259, 1.1344),
        "AutoencoderKL_F16": (-20.9596, 1.5877),
        "AutoencoderKL_F32": (-6.0891, 2.2547),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

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
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = (new_img ** (1 / scaler)) + miner

        return new_img


class WorkerQuantLogistics(WorkerQuantInterface):
    """Класс для логистического квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (0.055, 256.585),
        "AutoencoderVQ_F4": (-0.0138, 266.3559),
        "AutoencoderVQ_F8": (0.2718, 265.6474),
        "AutoencoderVQ_F16": (0.0622, 271.3231),
        "AutoencoderKL_F4": (-0.5459, 255.0000),
        "AutoencoderKL_F8": (1.8803, 255.0000),
        "AutoencoderKL_F16": (-0.3893, 255.0000),
        "AutoencoderKL_F32": (-0.0067, 255.9682),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner = self.quant_params[0]
        else:
            meaner = new_img.mean().item()

        new_img -= meaner
        new_img = 1 / (1 + np.exp(-new_img))
        new_max = np.max(new_img).item()

        if self._hardcore:
            scaler = self.quant_params[1]
        else:
            scaler = 255 / new_max
        new_img *= scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img /= scaler
        new_img = -np.log((1 / new_img) - 1)
        new_img += miner

        return new_img


class WorkerQuantMinLogistics(WorkerQuantInterface):
    """Класс для модифицированного (min вместо mean) логистического квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (-2.41, 256.585),
        "AutoencoderVQ_F4": (-4.0751, 255.2463),
        "AutoencoderVQ_F8": (-2.6578, 255.5716),
        "AutoencoderVQ_F16": (-2.5526, 256.2285),
        "AutoencoderKL_F4": (-67.7418, 255.0000),
        "AutoencoderKL_F8": (-31.1259, 255.0000),
        "AutoencoderKL_F16": (-20.9596, 255.0000),
        "AutoencoderKL_F32": (-6.0891, 255.0000),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner = self.quant_params[0]
        else:
            miner = new_img.min().item()

        new_img -= miner
        new_img = 1 / (1 + np.exp(-new_img))
        new_max = np.max(new_img).item()

        if self._hardcore:
            scaler = self.quant_params[1]
        else:
            scaler = 255 / new_max
        new_img *= scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img /= scaler
        new_img = -np.log((1 / new_img) - 1)
        new_img += miner

        return new_img


class WorkerQuantOddPower(WorkerQuantInterface):
    """Класс для нечётностепенного квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        3: {
            "default": (0.055, 8.16),
            "AutoencoderVQ_F4": (-0.0138, 1.8860),
            "AutoencoderVQ_F8": (0.2718, 6.8617),
            "AutoencoderVQ_F16": (0.0622, 7.9121),
            "AutoencoderKL_F4": (-0.5459, 0.0004),
            "AutoencoderKL_F8": (1.8803, 0.0077),
            "AutoencoderKL_F16": (-0.3893, 0.0146),
            "AutoencoderKL_F32": (-0.0067, 0.6250),
        },
        5: {
            "default": (0.055, 1.3056),
            "AutoencoderVQ_F4": (-0.0138, 0.1138),
            "AutoencoderVQ_F8": (0.2718, 0.9865),
            "AutoencoderVQ_F16": (0.0622, 1.2719),
            "AutoencoderKL_F4": (-0.5459, 0.00000010339),
            "AutoencoderKL_F8": (1.8803, 0.000015575),
            "AutoencoderKL_F16": (-0.3893, 0.000036621),
            "AutoencoderKL_F32": (-0.0067, 0.0198),
        }
    }

    def __init__(self, power: Union[int, float, str] = 3, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.power = int(power)
        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / (2*abs(new_img.min().item())**self.power)

        new_img -= meaner
        new_img = scaler*(new_img**self.power) + 127.5
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = ((new_img - 127.5) / scaler)
        mask = new_img < 0
        new_img = np.abs(new_img) ** (1/self.power)
        new_img[mask] *= -1
        new_img += meaner

        return new_img

    def adjust_params(self, autoencoder_worker: str = "default") -> tuple:
        """Настроить параметры под конкретный вариационный автокодировщик."""

        power_dict = self.quant_params_dict[self.power]
        new_params = power_dict.get(autoencoder_worker, None)
        if not new_params:
            new_params = power_dict["default"]

        self.quant_params = new_params
        self._hardcore = True

        return self.quant_params


class WorkerQuantTanh(WorkerQuantInterface):
    """Класс для тангенсуального квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (0.055, 127.5),
        "AutoencoderVQ_F4": (-0.0138, 127.5),
        "AutoencoderVQ_F8": (0.2718, 127.5),
        "AutoencoderVQ_F16": (0.0622, 127.5),
        "AutoencoderKL_F4": (-0.5459, 127.5),
        "AutoencoderKL_F8": (1.8803, 127.5),
        "AutoencoderKL_F16": (-0.3893, 127.5),
        "AutoencoderKL_F32": (-0.0067, 127.5),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / 2

        new_img -= meaner
        new_img = (np.tanh(new_img) + 1)*scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = np.arctanh(new_img / scaler - 1)
        new_img += meaner

        return new_img


class WorkerQuantMinTanh(WorkerQuantInterface):
    """Класс для модифицированного (mean -> min) тангенсуального квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (-2.41, 127.5),  # 253/2
        "AutoencoderVQ_F4": (-4.0751, 127.5),
        "AutoencoderVQ_F8": (-2.6578, 127.5),
        "AutoencoderVQ_F16": (-2.5526, 127.5),
        "AutoencoderKL_F4": (-67.7418, 127.5),
        "AutoencoderKL_F8": (-31.1259, 127.5),
        "AutoencoderKL_F16": (-20.9596, 127.5),
        "AutoencoderKL_F32": (-6.0891, 127.5),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner = new_img.min().item()
            scaler = 255 / 2

        new_img -= miner
        new_img = (np.tanh(new_img) + 1)*scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = new_img / scaler - 1
        new_img[new_img == 1.0] = 0.999
        new_img = np.arctanh(new_img)
        new_img += meaner

        return new_img


class WorkerQuantDoubleLogistics(WorkerQuantInterface):
    """Класс для двойного логистического квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (0.055, 255/2),
        "AutoencoderVQ_F4": (-0.0138, 127.5),
        "AutoencoderVQ_F8": (0.2718, 127.5),
        "AutoencoderVQ_F16": (0.0622, 127.5),
        "AutoencoderKL_F4": (-0.5459, 127.5),
        "AutoencoderKL_F8": (1.8803, 127.5),
        "AutoencoderKL_F16": (-0.3893, 127.5),
        "AutoencoderKL_F32": (-0.0067, 127.5),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / 2

        new_img -= meaner
        new_img = (np.sign(new_img)*(1 - np.exp(-(new_img**2))) + 1) * scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        # new_img = -torch.log(1 - ((new_img / scaler - 1) / torch.sign(new_img)))
        new_img = np.abs(1 - (new_img / scaler - 1))
        new_img[new_img == 0] = 0.005
        new_img = -np.log(new_img)
        mask = new_img < 0
        new_img = np.abs(new_img)
        new_img = np.sqrt(new_img)
        new_img[mask] *= -1
        new_img += meaner

        return new_img


class WorkerQuantMinDoubleLogistics(WorkerQuantInterface):
    """Класс для модифицированного (mean -> min) двойного логистического квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (-2.41, 255/2),
        "AutoencoderVQ_F4": (-4.0751, 127.5),
        "AutoencoderVQ_F8": (-2.6578, 127.5),
        "AutoencoderVQ_F16": (-2.5526, 127.5),
        "AutoencoderKL_F4": (-67.7418, 127.5),
        "AutoencoderKL_F8": (-31.1259, 127.5),
        "AutoencoderKL_F16": (-20.9596, 127.5),
        "AutoencoderKL_F32": (-6.0891, 127.5),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner = new_img.min().item()
            scaler = 255 / 2

        new_img -= miner
        new_img = (np.sign(new_img)*(1 - np.exp(-(new_img**2))) + 1) * scaler
        new_img = np.round(new_img)
        new_img = new_img.clip(0, 255)
        new_img = new_img.astype(np.uint8)

        quant_params = [miner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = np.abs(1 - (new_img / scaler - 1))
        new_img[new_img == 0] = 0.005
        new_img = -np.log(new_img)
        mask = new_img < 0
        new_img = np.abs(new_img)
        new_img = np.sqrt(new_img)
        new_img[mask] *= -1
        new_img += meaner

        return new_img


class WorkerQuantSinh(WorkerQuantInterface):
    """Класс для гиперболическосинусоидального квантования и деквантования с нормализированными параметрами."""

    quant_params_dict = {
        "default": (0.055, 21.0737),
        "AutoencoderVQ_F4": (-0.0138, 4.3391),
        "AutoencoderVQ_F8": (0.2718, 18.0781),
        "AutoencoderVQ_F16": (0.0622, 20.3450),
        "AutoencoderKL_F4": (-0.5459, 0.0000),
        "AutoencoderKL_F8": (1.8803, 0.0000),
        "AutoencoderKL_F16": (-0.3893, 0.0000),
        "AutoencoderKL_F32": (-0.0067, 0.7750),
    }

    def __init__(self, pre_quant: str = "", nsd: int = 0, hardcore: bool = True):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, np.float16 или np.float32."""

        self.pre_quant = pre_quant
        self.nsd = int(nsd)
        self.quant_params = []
        if hardcore:
            self.adjust_params()
        self._hardcore = hardcore

    def quant_work(self, new_img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Квантование np.ndarray из типа np.float16 в np.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self.pre_quant:
            new_img = self.pre_quantize(new_img)

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner = new_img.mean().item()
            scaler = 255 / (2*np.sinh(abs(new_img.min())).item())

        new_img -= meaner
        new_img = scaler*np.sinh(new_img) + 127.5
        new_img = np.round(new_img)
        new_img = new_img.astype(np.uint8)
        new_img = new_img.clip(0, 255)

        quant_params = [meaner, scaler]

        return new_img, quant_params

    def dequant_work(self, new_img: np.ndarray, dest_type=np.float16, params=None) -> np.ndarray:
        """Деквантование np.ndarray из типа np.uint8 в np.float.
        Вход: (квантованный тензор, тип данных, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор."""

        if self._hardcore:
            meaner, scaler = self.quant_params
        else:
            meaner, scaler = params

        new_img = new_img.astype(dest_type)
        new_img = np.arcsinh((new_img - 127.5) / scaler)
        new_img += meaner

        return new_img


class WorkerSRInterface:
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


class WorkerPredictorInterface:
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


if __name__ == "__main__":
    test = WorkerDummy()
    result = test.do_work()
    print(result)
