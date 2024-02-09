import copy
import math
import time
import zlib
from typing import Callable
from abc import abstractmethod
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from basicsr.archs.rrdbnet_arch import RRDBNet
from outer_models.util import instantiate_from_config
from outer_models.realesrgan import RealESRGANer
from outer_models.prediction.model.models import Model as Predictor, DMVFN


# WorkerMeta -- метакласс для декорации -> получения времени
# WorkerDummy -- класс ложного ("ленивого") рабочего, имитирующего деятельность

# > WorkerCompressorInterface -- абстрактный класс интерфейса для сжатия/расжатия
# WorkerCompressorDeflated -- класс рабочего для сжатия и расжатия Deflated

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


class WorkerCompressorDeflated(WorkerCompressorInterface):
    """Рабочий Deflated."""

    def __init__(self, level=9):
        self.level = level

        self._compressor = zlib.compressobj(level=self.level, method=zlib.DEFLATED)
        self._decompressor = zlib.decompressobj()

    def compress_work(self, latent_img: torch.Tensor) -> bytes:
        """Сжатие Deflated.
        Вход: картинка в виде torch.Tensor.
        Выход: bytes."""

        numpy_img = latent_img.numpy()
        byter = numpy_img.tobytes()

        new_min = self._compressor.compress(byter)
        new_min += self._compressor.flush()

        return new_min

    def decompress_work(self, compressed_bytes: bytes) -> torch.Tensor:
        """Расжатие Deflated.
        Вход: bytes.
        Выход: картинка в виде torch.Tensor."""

        byters = self._decompressor.decompress(compressed_bytes)
        byters += self._decompressor.flush()

        latent_img = torch.frombuffer(byters, dtype=torch.uint8)
        latent_img = latent_img.reshape(1, 8, 32, 32)

        return latent_img


class WorkerAutoencoderInterface(metaclass=WorkerMeta):
    """Интерфейс для рабочих-автокодировщиков."""

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
    """Рабочий VAE KL-f16."""

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
    def sr_work(self, img: np.ndarray) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray).
        Выход: изображение в формате cv2 (np.ndarray)."""

        pass


class WorkerSRDummy(WorkerSRInterface):
    """Ложный класс работника суперрезолюции."""

    def __init__(self, dest_height: int = 1080, dest_width: int = 1920):
        """dest_height -- высота результирующего изображения.
        dest_width -- ширина результирующего изображения."""

        self._dest_size = [dest_width // self.this_scale, dest_height // self.this_scale]

    def sr_work(self, img: np.ndarray) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray).
        Выход: изображение в формате cv2 (np.ndarray)."""

        new_img = cv2.resize(img, self._dest_size)

        return new_img


class WorkerSRRealESRGAN_x2plus(WorkerSRInterface):
    """Класс работника суперрезолюции с ESRGAN вариации Real x2."""

    this_scale = 2

    def __init__(self, path: str, dni_base: float = 0.75,
                 dest_height: int = 1080, dest_width: int = 1920):
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

    def sr_work(self, img: np.ndarray) -> np.ndarray:
        """Суперрезолюция изображения.
        Вход: изображение в формате cv2 (np.ndarray).
        Выход: изображение в формате cv2 (np.ndarray)."""

        new_img = cv2.resize(img, self._dest_size)
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
