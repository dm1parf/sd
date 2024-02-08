import time
import zlib
import math
from typing import Callable
import torch
from omegaconf import OmegaConf
from outer_models.util import instantiate_from_config

# WorkerMeta -- метакласс для декорации -> получения времени
# WorkerDummy -- класс ложного ("ленивого") рабочего, имитирующего деятельность

# WorkerDeflated -- класс рабочего для сжатия и расжатия Deflated

# WorkerLinear -- класс рабочего для линейного квантования и деквантования
# WorkerPower -- класс рабочего для степенного квантования и деквантования
# WorkerLogistics -- класс рабочего для логистического квантования и деквантования

# WorkerVQ_F16 -- класс рабочего вариационного автокодировщика VQ-f16
# WorkerKL_F16 -- класс рабочего вариационного автокодировщика KL-f16
# WorkerKL_F32 -- класс рабочего вариационного автокодировщика KL-f32


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
        def internal_func(strict_sync: bool = False, *args, **kwargs):
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


class WorkerDeflated(metaclass=WorkerMeta):
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


class WorkerVQ_F16(metaclass=WorkerMeta):
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


class WorkerKL_F16(metaclass=WorkerMeta):
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


class WorkerKL_F32(metaclass=WorkerMeta):
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


class WorkerLinear(metaclass=WorkerMeta):
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


class WorkerPower(metaclass=WorkerMeta):
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


class WorkerLogistics(metaclass=WorkerMeta):
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


if __name__ == "__main__":
    test = WorkerDummy()
    result = test.do_work()
    print(result)
