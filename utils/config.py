import configparser
from typing import Optional
from utils.statistics import StatisticsManager
from utils.uav_dataset import UAVDataset
from utils.workers import *


class ConfigManager:
    config_encoding = 'utf-8'

    autoencoder_types = {
        "AutoencoderVQ_F16": WorkerAutoencoderVQ_F16,
        "AutoencoderKL_F16": WorkerAutoencoderKL_F16,
        "AutoencoderKL_F32": WorkerAutoencoderKL_F32,
    }

    quantizer_types = {
        "QuantLinear": WorkerQuantLinear,
        "QuantPower": WorkerQuantPower,
        "QuantLogistics": WorkerQuantLogistics,
    }

    compressor_types = {
        "CompressorDummy": WorkerCompressorDummy,
        "CompressorDeflated": WorkerCompressorDeflated,
        "CompressorLzma": WorkerCompressorLzma,
        "CompressorGzip": WorkerCompressorGzip,
        "CompressorBzip2": WorkerCompressorBzip2,
        "CompressorH264": WorkerCompressorH264,
        "CompressorH265": WorkerCompressorH265,
    }

    sr_types = {
        "SRDummy": WorkerSRDummy,
        "SRRealESRGAN_x2plus": WorkerSRRealESRGAN_x2plus,
    }

    predictor_types = {
        "PredictorDummy": WorkerPredictorDummy,
        "PredictorDMVFN": WorkerPredictorDMVFN,
    }

    def __init__(self, config_path: str):
        """config_path -- путь к INI-файлу конфигурации эксперимента."""

        self._config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(self._config_path, encoding=self.config_encoding)

        self._common_settings = self.config["CommonSettings"]
        self._autoenc_settings = self.config["AutoencoderSettings"]
        self._quant_settings = self.config["QuantizerSettings"]
        self._compress_settings = self.config["CompressorSettings"]
        self._sr_settings = self.config["SRSettings"]
        self._predictor_settings = self.config["PredictorSettings"]

    # Общие настройки

    def get_stat_mng(self) -> StatisticsManager:
        """Получить имя файла для сохранения статистики."""

        stat_filename = self._common_settings["stat_filename"]
        stat_mng = StatisticsManager(stat_filename)
        return stat_mng

    def get_dataset(self) -> UAVDataset:
        """Получить путь к набору данных."""

        dataset_path = self._common_settings["dataset_path"]
        dataset = UAVDataset(dataset_path, name_output=True)
        return dataset

    def get_max_entries(self) -> int:
        """Получить максимальное количество картинок для обработки.
        0 значит без ограничений."""

        max_entries = int(self._common_settings["max_entries"])
        return max_entries

    def get_progress_check(self) -> int:
        """Получить количество картинок до отображения прогресса."""

        progress_check = int(self._common_settings["progress_check"])
        return progress_check

    def get_imwrite_params(self) -> tuple[int, str]:
        """Получить параметры записи изображений."""

        image_write = int(self._common_settings["image_write"])
        image_write_path = self._common_settings["image_write_path"]
        return image_write, image_write_path

    def get_autoencoder_worker(self) -> Optional[WorkerAutoencoderInterface]:
        """Получить рабочий-автокодировщик из настроек."""

        use_autoencoder = bool(int(self._autoenc_settings["use_autoencoder"]))
        if not use_autoencoder:
            return None
        autoencoder_type = self._autoenc_settings["autoencoder_type"].strip()
        config_path = self._autoenc_settings["config_path"]
        ckpt_path = self._autoenc_settings["ckpt_path"]
        if autoencoder_type in self.autoencoder_types:
            new_autoencoder = self.autoencoder_types[autoencoder_type](config_path=config_path, ckpt_path=ckpt_path)
        else:
            raise NotImplementedError("Неподдерживаемый тип автокодировщика:", autoencoder_type)

        return new_autoencoder

    def get_quant_worker(self) -> Optional[WorkerQuantInterface]:
        """Получить рабочий квантования из настроек."""

        use_quantizer = bool(int(self._quant_settings["use_quantizer"]))
        if not use_quantizer:
            return None
        quantizer_type = self._quant_settings["quantizer_type"].strip()
        if quantizer_type in self.quantizer_types:
            new_quantizer = self.quantizer_types[quantizer_type]()
        else:
            raise NotImplementedError("Неподдерживаемый тип квантовальщика:", quantizer_type)
        return new_quantizer

    def get_compress_worker(self) -> WorkerCompressorInterface:
        """Получить рабочий сжатия из настроек."""

        compressor_type = self._compress_settings["compressor_type"].strip()
        if compressor_type in self.compressor_types:
            new_compressor = self.compressor_types[compressor_type]()
        else:
            raise NotImplementedError("Неподдерживаемый тип сжатия:", compressor_type)
        return new_compressor

    def get_sr_worker(self) -> WorkerSRInterface:
        """Получить рабочий сверхразрешения из настроек."""

        sr_type = self._sr_settings["sr_type"].strip()
        if sr_type in self.sr_types:
            config_path = self._sr_settings["config_path"]
            if ("dest_height" in self._sr_settings) and ("dest_width" in self._sr_settings):
                dest_height = self._sr_settings["dest_height"]
                dest_width = self._sr_settings["dest_width"]
                new_sr = self.sr_types[sr_type](config_path, dest_height=dest_height, dest_width=dest_width)
            else:
                self.sr_types[sr_type](config_path)
        else:
            raise NotImplementedError("Неподдерживаемый тип сжатия:", sr_type)
        return new_sr

    def get_predictor_worker(self) -> Optional[WorkerPredictorInterface]:
        """Получить рабочий предиктора из настроек."""

        use_predictor = bool(int(self._predictor_settings["use_predictor"]))
        if not use_predictor:
            return None
        predictor_type = self._predictor_settings["predictor_type"].strip()
        if predictor_type in self.predictor_types:
            config_path = self._predictor_settings["config_path"]
            new_predictor = self.predictor_types[predictor_type](config_path)
        else:
            raise NotImplementedError("Неподдерживаемый тип предиктора:", predictor_type)
        return new_predictor
