import copy
import math
import time
import zlib
import lzma
import bz2
import gzip
import io

from typing import Callable
from abc import abstractmethod
import cv2
import torch
import imageio
import numpy as np
from omegaconf import OmegaConf
from basicsr.archs.rrdbnet_arch import RRDBNet
from dependence.util import instantiate_from_config
from dependence.realesrgan import RealESRGANer
from dependence.prediction.model.models import Model as Predictor, DMVFN


# WorkerMeta -- метакласс для декорации -> получения времени
# WorkerDummy -- класс ложного ("ленивого") рабочего, имитирующего деятельность

# > WorkerCompressorInterface -- абстрактный класс интерфейса для сжатия/расжатия
# WorkerCompressorDummy -- класс ложного ("ленивого") рабочего, имитирующего сжатие
# WorkerCompressorDeflated -- класс рабочего для сжатия и расжатия Deflated
# WorkerCompressorLzma -- класс рабочего для сжатия и расжатия Lzma
# WorkerCompressorGzip -- класс рабочего для сжатия и расжатия Gzip
# WorkerCompressorBzip2 -- класс рабочего для сжатия и расжатия Bzip2
# WorkerCompressorH264 -- класс рабочего для сжатия и расжатия Bzip2
# WorkerCompressorH265 -- класс рабочего для сжатия и расжатия Bzip2

# > WorkerAutoencoderInterface -- абстрактный класс интерфейса для автокодировщиков
# WorkerAutoencoderVQ_F16 -- класс рабочего вариационного автокодировщика VQ-f16
# WorkerAutoencoderKL_F16 -- класс рабочего вариационного автокодировщика KL-f16
# WorkerAutoencoderKL_F32 -- класс рабочего вариационного автокодировщика KL-f32

# > WorkerQuantInterface -- абстрактный класс интерфейса для квантования
# WorkerQuantLinear -- класс рабочего для линейного квантования и деквантования
# WorkerQuantPower -- класс рабочего для степенного квантования и деквантования
# WorkerQuantLogistics -- класс рабочего для логистического квантования и деквантования

# > WorkerSRInterface -- абстрактный класс интерфейса для суперрезолюции
# WorkerSRDummy -- класс ложного ("ленивого") рабочего, имитирующего суперрезолюцию
# WorkerSRRealESRGAN_x2plus -- класс рабочего SR вида ESRGAN Plus x2

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
        def internal_func(*args, strict_sync: bool = False, **kwargs):
            if strict_sync:
                torch.cuda.synchronize()
            start = time.time()
            result = func(*args, **kwargs)
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
        image = image.reshape(3, 512, 512)
        image = image.cpu()
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        image = image.reshape(1, 512, 512, 3)

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

    def __init__(self, device='cuda', *_, **__):
        self.device = device

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие H265.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        latent_img *= 255.0
        image = latent_img.to(torch.uint8)
        image = image.reshape(3, 512, 512)
        image = image.cpu()
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        image = image.reshape(1, 512, 512, 3)

        buffer = io.BytesIO()
        writer = imageio.get_writer(buffer, format="mov", codec="hevc", fps=30)
        writer.append_data(image)
        try:  # Иногда падает
            writer.close()
        except EOFError:
            pass
        buffer.seek(0, 0)
        new_min = buffer.read()

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


class WorkerAutoencoderInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих-автокодировщиков."""

    z_shape = (1, 0, 0, 0)

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
        self._model = self._model.type(torch.float16).cuda()

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


class WorkerAutoencoderKL_F16(WorkerAutoencoderInterface):
    """Рабочий VAE KL-f16."""

    z_shape = (1, 16, 32, 32)

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
        self._model = self._model.type(torch.float16).cuda()

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        gauss = self._model.encode(from_image)
        latent = gauss.sample()
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        to_image = self._model.decode(latent)
        return to_image


class WorkerAutoencoderKL_F32(WorkerAutoencoderInterface):
    """Рабочий VAE KL-f32."""

    z_shape = (1, 64, 16, 16)

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
        self._model = self._model.type(torch.float16).cuda()

    def encode_work(self, from_image: torch.Tensor) -> torch.Tensor:
        """Кодирование картинки в латентное пространство.
        Вход: картинка в виде torch.Tensor.
        Выход: латентное пространство в виде torch.Tensor."""

        gauss = self._model.encode(from_image)
        latent = gauss.sample()
        return latent

    def decode_work(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование картинки в латентное пространство.
        Вход: латентное пространство в виде torch.Tensor.
        Выход: картинка в виде torch.Tensor."""

        to_image = self._model.decode(latent)
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
    def dequant_work(self, latent: torch.Tensor, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор"""

        pass


class WorkerQuantLinear(WorkerQuantInterface):
    """Класс для линейного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, hardcore: bool = True, dest_type=torch.float16):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 47.69)
        self._hardcore = hardcore
        self.dest_type = dest_type

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            maxer = latent.max().item()
            miner = latent.min().item()
            aller = maxer - miner
            scaler = 255 / aller

        new_img = torch.clone(latent)
        new_img = (new_img - miner) * scaler
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор"""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(self.dest_type)
        new_img = (new_img / scaler) + miner

        return new_img


class WorkerQuantPower(WorkerQuantInterface):
    """Класс для линейного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, hardcore: bool = True, dest_type=torch.float16):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 3.31)
        self._hardcore = hardcore
        self.dest_type = dest_type

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            maxer = latent.max().item()
            miner = latent.min().item()
            aller = maxer - miner
            scaler = math.log(255, aller)

        new_img = torch.clone(latent)
        new_img = (new_img - miner) ** scaler
        new_img = new_img.to(torch.uint8)
        new_img = new_img.clamp(0, 255)

        quant_params = (miner, scaler)
        return new_img, quant_params

    def dequant_work(self, latent: torch.Tensor, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор"""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(self.dest_type)
        new_img = (new_img ** (1 / scaler)) + miner

        return new_img


class WorkerQuantLogistics(WorkerQuantInterface):
    """Класс для линейного квантования и деквантования с нормализированными параметрами."""

    def __init__(self, hardcore: bool = True, dest_type=torch.float16):
        """hardcore -- использование жёстко заданных параметров квантования и деквантования.
        dest_type -- результирующий тип. Например, torch.float16 или torch.float32."""

        self.quant_params = []
        if hardcore:
            self.quant_params = (-2.41, 256.585)
        self._hardcore = hardcore
        self.dest_type = dest_type

    def quant_work(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        """Квантование torch.Tensor из типа torch.float в torch.uint8.
        Вход: (деквантованный тензор)
        Выход: (квантованный тензор, (параметр сдвига, параметр масштабирования))"""

        if self._hardcore:  # Так не нужно передавать, но в зависимости от картинки хуже
            miner = self.quant_params[0]
        else:
            miner = latent.min().item()

        new_img = torch.clone(latent)
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

    def dequant_work(self, latent: torch.Tensor, params=None) -> torch.Tensor:
        """Деквантование torch.Tensor из типа torch.uint8 в torch.float.
        Вход: (квантованный тензор, опционально (параметр сдвига, параметр масштабирования))
        Выход: деквантованный тензор"""

        if self._hardcore:
            miner, scaler = self.quant_params
        else:
            miner, scaler = params

        new_img = torch.clone(latent)
        new_img = new_img.to(self.dest_type)
        new_img /= scaler
        new_img = -torch.log((1 / new_img) - 1)
        new_img += miner

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

    def __init__(self, path: str = "", dest_height: int = 720, dest_width: int = 1280):
        """dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        self._dest_size = [dest_width, dest_height]

    def sr_work(self, img: np.ndarray, dest_size: list = None) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray), dest_size (опционально) -- новый размер.
        Выход: изображение в формате cv2 (np.ndarray)."""

        if not dest_size:
            dest_size = self._dest_size
        new_img = cv2.resize(img, dest_size)

        return new_img


class WorkerSRRealESRGAN_x2plus(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 2

    def __init__(self, path: str,
                 dni_base: float = 0.75, dest_height: int = 720, dest_width: int = 1280):
        """path -- путь к pth-файлу весов модели.
        dni_base -- основной уровень шума (0-1).
        dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""
        self._backend_model = backend_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                                      num_block=23, num_grow_ch=32, scale=self.this_scale)

        dni_base = dni_base
        dni_weight = [dni_base, 1 - dni_base]
        self._model = RealESRGANer(scale=self.this_scale, model_path=path, dni_weight=dni_weight, tile=0, tile_pad=10,
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
            images *= 2
        predict_images = self._model.predict(images, predict_num)

        if not isinstance(predict_images, list):
            predict_img = [predict_images]

        return predict_images


if __name__ == "__main__":
    test = WorkerDummy()
    result = test.do_work()
    print(result)
