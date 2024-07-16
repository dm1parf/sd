import configparser
from typing import Optional

import torch.utils.data

from utils.planner import ExperimentPlanner
from utils.statistics import StatisticsManager
from utils.uav_dataset import UAVDataset
from utils.workers import *


class ConfigManager:
    config_encoding = 'utf-8'

    section_names = {
        "experiment_settings": "ExperimentSettings",
        "as_settings": "ASSettings",
        "autoencoder_settings": "AutoencoderSettings",
        "quantizer_settings": "QuantizerSettings",
        "compressor_settings": "CompressorSettings",
        "sr_settings": "SRSettings",
        "predictor_settings": "PredictorSettings",
        "planner_settings": "PlannerSettings",
    }

    as_types = {
        "ASDummy": WorkerASDummy,
        "ASCutEdgeColors": WorkerASCutEdgeColors,
        "ASMoveDistribution": WorkerASMoveDistribution,
        "ASComposit": WorkerASComposit,
    }

    autoencoder_types = {
        "AutoencoderVQ_F4": WorkerAutoencoderVQ_F4,
        "AutoencoderVQ_F8": WorkerAutoencoderVQ_F8,
        "AutoencoderVQ_F16": WorkerAutoencoderVQ_F16,
        "AutoencoderVQ_F16_Optimized": WorkerAutoencoderVQ_F16_Optimized,
        "AutoencoderKL_F4": WorkerAutoencoderKL_F4,
        "AutoencoderKL_F8": WorkerAutoencoderKL_F8,
        "AutoencoderKL_F16": WorkerAutoencoderKL_F16,
        "AutoencoderKL_F32": WorkerAutoencoderKL_F32,
        "AutoencoderCDC": WorkerAutoencoderCDC,
    }

    quantizer_types = {
        "QuantLinear": WorkerQuantLinear,
        "QuantPower": WorkerQuantPower,
        "QuantLogistics": WorkerQuantLogistics,
        "QuantMinLogistics": WorkerQuantMinLogistics,
        "QuantOddPower": WorkerQuantOddPower,
        "QuantTanh": WorkerQuantTanh,
        "QuantMinTanh": WorkerQuantMinTanh,
        "QuantDoubleLogistics": WorkerQuantDoubleLogistics,
        "QuantMinDoubleLogistics": WorkerQuantMinDoubleLogistics,
        "QuantSinh": WorkerQuantSinh,
    }

    compressor_types = {
        "CompressorDummy": WorkerCompressorDummy,
        "CompressorDeflated": WorkerCompressorDeflated,
        "CompressorLzma": WorkerCompressorLzma,
        "CompressorGzip": WorkerCompressorGzip,
        "CompressorBzip2": WorkerCompressorBzip2,
        "CompressorZstd": WorkerCompressorZstd,
        "CompressorBrotli": WorkerCompressorBrotli,
        "CompressorLz4": WorkerCompressorLz4,
        "CompressorLz4f": WorkerCompressorLz4f,
        "CompressorLz4h5": WorkerCompressorLz4h5,
        "CompressorLzw": WorkerCompressorLzw,
        "CompressorLzf": WorkerCompressorLzf,
        "CompressorLzfse": WorkerCompressorLzfse,
        "CompressorAec": WorkerCompressorAec,
        "CompressorH264": WorkerCompressorH264,
        "CompressorH265": WorkerCompressorH265,
        "CompressorJpeg": WorkerCompressorJpeg,
        "CompressorAvif": WorkerCompressorAvif,
        "CompressorHeic": WorkerCompressorHeic,
        "CompressorWebp": WorkerCompressorWebp,
        "CompressorJpegLS": WorkerCompressorJpegLS,
        "CompressorJpegXR": WorkerCompressorJpegXR,
        "CompressorJpegXL": WorkerCompressorJpegXL,
        "CompressorQoi": WorkerCompressorQoi,
    }

    sr_types = {
        "SRDummy": WorkerSRDummy,
        "SRRealESRGAN_x2plus": WorkerSRRealESRGAN_x2plus,
        "SRAPISR_RRDB_x2": WorkerSRAPISR_RRDB_x2,
        "SRAPISR_RRDB_x2_Optimized": WorkerSRAPISR_RRDB_x2_Optimized,
        "SRAPISR_GRL_x4": WorkerSRAPISR_GRL_x4,
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

        if self.section_names["experiment_settings"] in self.config:
            self._common_settings = self.config[self.section_names["experiment_settings"]]
        else:
            self._common_settings = None
        if self.section_names["as_settings"] in self.config:
            self._as_settings = self.config[self.section_names["as_settings"]]
        else:
            self._as_settings = None
        if self.section_names["autoencoder_settings"] in self.config:
            self._autoenc_settings = self.config[self.section_names["autoencoder_settings"]]
        else:
            self._autoenc_settings = None
        if self.section_names["quantizer_settings"] in self.config:
            self._quant_settings = self.config[self.section_names["quantizer_settings"]]
        else:
            self._quant_settings = None
        if self.section_names["compressor_settings"] in self.config:
            self._compress_settings = self.config[self.section_names["compressor_settings"]]
        else:
            self._common_settings = None
        if self.section_names["sr_settings"] in self.config:
            self._sr_settings = self.config[self.section_names["sr_settings"]]
        else:
            self._sr_settings = None
        if self.section_names["predictor_settings"] in self.config:
            self._predictor_settings = self.config[self.section_names["predictor_settings"]]
        else:
            self._predictor_settings = None
        if self.section_names["planner_settings"] in self.config:
            self._planner_settings = self.config[self.section_names["planner_settings"]]
        else:
            self._planner_settings = None

    # Общие настройки

    def get_stat_mng(self) -> StatisticsManager:
        """Получить имя файла для сохранения статистики."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        stat_filename = self._common_settings["stat_filename"]
        rounder = self._common_settings["round_digits"]
        min_site_type = self._common_settings["min_site_type"]
        if rounder:
            rounder = int(rounder)
        else:
            rounder = None

        stat_mng = StatisticsManager(stat_filename, rounder=rounder, size_type=min_site_type)
        return stat_mng

    def get_dataset(self) -> UAVDataset:
        """Получить путь к набору данных."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        dataset_path = self._common_settings["dataset_path"]
        dataset_shuffle = self._common_settings.getboolean("dataset_shuffle")
        dataset = UAVDataset(dataset_path, name_output=True, shuffle=dataset_shuffle)
        return dataset

    def get_basic_size(self) -> Optional[tuple[int, int]]:
        """Получить базовый размер набора."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        basic_width = self._common_settings["basic_width"]
        basic_height = self._common_settings["basic_height"]

        if basic_width and basic_height:
            basic_width = int(basic_width)
            basic_height = int(basic_height)
            return basic_width, basic_height
        else:
            return None

    def get_data_loader(self) -> torch.utils.data.DataLoader:
        """Получить путь к набору данных."""

        dataset = self.get_dataset()
        batch_size = int(self._common_settings["batch_size"])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return data_loader

    def get_max_entries(self) -> int:
        """Получить максимальное количество картинок для обработки.
        0 значит без ограничений."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        max_entries = int(self._common_settings["max_entries"])
        return max_entries

    def get_progress_check(self) -> int:
        """Получить количество картинок до отображения прогресса."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        progress_check = int(self._common_settings["progress_check"])
        return progress_check

    def get_imwrite_params(self) -> tuple[int, str]:
        """Получить параметры записи изображений."""

        if not self._common_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["experiment_settings"]))

        image_write = int(self._common_settings["image_write"])
        image_write_path = self._common_settings["image_write_path"]
        return image_write, image_write_path

    def get_as_worker(self) -> WorkerASInterface:
        """Получить рабочий подавителя артефактов из настроек."""

        if not self._as_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["as_settings"]))

        as_type = self._as_settings["as_type"].strip()
        if as_type in self.as_types:
            params = self._as_settings["as_params"].split()
            new_as = self.as_types[as_type](*params)
        else:
            raise NotImplementedError("Неподдерживаемый тип подавителя артефактов:", as_type)
        return new_as

    def get_autoencoder_worker(self) -> Optional[WorkerAutoencoderInterface]:
        """Получить рабочий-автокодировщик из настроек."""

        if not self._autoenc_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["autoencoder_settings"]))

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

        if not self._quant_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["quantizer_settings"]))

        use_quantizer = bool(int(self._quant_settings["use_quantizer"]))
        if not use_quantizer:
            return None
        quantizer_type = self._quant_settings["quantizer_type"].strip()
        if quantizer_type in self.quantizer_types:
            params = self._quant_settings["quantizer_params"].split()
            new_quantizer = self.quantizer_types[quantizer_type](*params)

            if self._autoenc_settings:
                use_autoencoder = bool(int(self._autoenc_settings["use_autoencoder"]))
                if use_autoencoder:
                    autoencoder_type = self._autoenc_settings["autoencoder_type"].strip()
                    new_quantizer.adjust_params(autoencoder_type)

        else:
            raise NotImplementedError("Неподдерживаемый тип квантовальщика:", quantizer_type)
        return new_quantizer

    def get_compress_worker(self) -> WorkerCompressorInterface:
        """Получить рабочий сжатия из настроек."""

        if not self._compress_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["compressor_settings"]))

        compressor_type = self._compress_settings["compressor_type"].strip()
        if compressor_type in self.compressor_types:
            params = self._compress_settings["compressor_params"].split()
            new_compressor = self.compressor_types[compressor_type](*params)
        else:
            raise NotImplementedError("Неподдерживаемый тип сжатия:", compressor_type)
        return new_compressor

    def get_sr_worker(self) -> WorkerSRInterface:
        """Получить рабочий сверхразрешения из настроек."""

        if not self._sr_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["sr_settings"]))

        sr_type = self._sr_settings["sr_type"].strip()
        if sr_type in self.sr_types:
            config_path = self._sr_settings["config_path"]
            ckpt_path = self._sr_settings["ckpt_path"]

            if ("dest_height" in self._sr_settings) and ("dest_width" in self._sr_settings):
                dest_height = int(self._sr_settings["dest_height"])
                dest_width = int(self._sr_settings["dest_width"])
                new_sr = self.sr_types[sr_type](config_path, ckpt_path, dest_height=dest_height, dest_width=dest_width)
            else:
                self.sr_types[sr_type](config_path, ckpt_path)
        else:
            raise NotImplementedError("Неподдерживаемый тип сверхразрешения:", sr_type)
        return new_sr

    def get_predictor_worker(self) -> Optional[WorkerPredictorInterface]:
        """Получить рабочий предиктора из настроек."""

        if not self._predictor_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["predictor_settings"]))

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

    def get_planner(self) -> ExperimentPlanner:
        """Получить планировщик экспериментов."""

        if not self._planner_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["planner_settings"]))

        sections_path = self._planner_settings["sections_path"]
        experiment_config_path = self._planner_settings["experiment_config_path"]
        experiment_module = self._planner_settings["experiment_module"]
        statistics_path = self._planner_settings["statistics_path"]
        index_filename = self._planner_settings["index_filename"]

        planner_config = [sections_path, experiment_config_path, experiment_module]
        if statistics_path:
            planner_config.append(statistics_path)
        if index_filename:
            planner_config.append(index_filename)

        planner = ExperimentPlanner(*planner_config)
        return planner

    def get_start_experiment(self) -> Optional[int]:
        """Получить начальный эксперимент для планировщика."""

        if not self._planner_settings:
            raise configparser.NoSectionError("No section \"{}\"!".format(self.section_names["planner_settings"]))

        try:
            start_experiment = self._planner_settings.getint("start_experiment")
            return start_experiment
        except:
            return None
