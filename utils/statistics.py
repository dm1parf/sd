import os
import csv
import math
from typing import Union, Sequence, Mapping, Optional, Literal
import scipy
import pandas as pd
import numpy as np
import cv2
from skimage.metrics import structural_similarity


class StatisticsManager:
    """Класс для записи статистических данных в csv."""

    default_filename = "statistics.csv"
    summary_definer = "_summary"
    nominal_types = [int, str,
                     float, float, float,
                     int,
                     float, float, float,
                     float, float,
                     float, float,
                     float, float,
                     float, float,
                     float, float,
                     float, float, float,
                     float]
    stat_params = ["id", "name",
                   "psnr", "mse", "ssim",
                   "nuniq",
                   "latent_size", "min_size", "latent_compression_ratio",
                   "as_prepare_time", "as_restore_time",
                   "encoding_time", "decoding_time",
                   "quant_time", "dequant_time",
                   "compress_time", "decompress_time",
                   "superresolution_time", "predictor_time",
                   "total_coder_time", "total_decoder_time", "total_time",
                   "bitrate"]
    # id -- идентификатор испытания.
    # name -- имя файла.
    # psnr -- метрика PSNR.
    # mse -- метрика MSE.
    # ssim -- метрика SSIM.
    # nuniq -- число уникальных значений после квантования.
    # latent_size -- размер латентного пространства до сжатия.
    # min_size -- минимальный размер в сжатом виде в байтах или Кб.
    # latent_compression_ratio -- степень сжатия латентного пространства.
    # as_prepare_time -- время подготовки подавления артефактов.
    # as_restore_time -- время восстановления после подавления артефактов.
    # encoding_time -- время кодирования.
    # decoding_time -- время декодирования.
    # quant_time -- время квантования.
    # dequant_time -- время деквантования.
    # compress_time -- время сжатия.
    # decompress_time -- время расжатия.
    # superresolution_time -- время суперрезолюции.
    # predictor_time -- время работы предиктора.
    # total_coder_time -- полное время работы пайплайна на стороне устройства-кодера.
    # total_decoder_time -- полное время работы пайплайна на стороне устройства-декодера.
    # total_time -- полное время проведения испытания.
    # bitrate -- полное время проведения испытания.

    summary_params = ["value", "count",
                      "mean", "med",
                      "abs_min", "abs_max",
                      "var", "std",
                      "conf_p", "conf", "conf_min", "conf_max"]
    # value -- величина.
    # count -- всего значений.
    # mean -- среднее.
    # med -- медиана.
    # abs_min -- абсолютный минимум.
    # abs_max -- абсолютный максимум.
    # var -- дисперсия.
    # std -- среднее.
    # conf_p -- доверительная довероятность.
    # conf -- половина длины доверительного интервала.
    # conf_min -- минимум с учётом доверительного интервала.
    # conf_max -- максимум с учётом доверительного интервала.

    # +++ black_frame_rate -- доля чёрных кадров в итоговой статистике.

    def __init__(self, filename: str = None, placeholder="-", rounder: Optional[int] = None,
                 size_type: Literal["b", "Kb"] = "b"):
        self.filename = ""
        self.placeholder = placeholder
        self.rounder = rounder
        self.size_type = size_type
        if self.size_type != "b":
            self._size_indexes = []
            latent_size_index = self.stat_params.index("latent_size")
            min_size_index = self.stat_params.index("min_size")
            self._size_indexes.append(latent_size_index)
            self._size_indexes.append(min_size_index)

        if not filename:
            filename = self.default_filename
        self._file = None
        self._csv = None
        self.data = []  # Явно сохраняем данные для дальнейшего анализа.

        self.change_file(filename)

    # Некоторые технические методы

    def _round_data(self, new_seq: list):
        """Округление данных."""

        if self.rounder:
            for i, (dater, typer) in enumerate(zip(new_seq, self.nominal_types)):
                if (typer is float) and isinstance(dater, float):
                    new_seq[i] = round(dater, self.rounder)

        return new_seq

    def _write_in_stat_file(self, csv_file, base_file, row: Union[Sequence, Mapping], header=False) -> None:
        """Запись в какой-нибудь файл с выталкиванием буфера."""

        if isinstance(row, Sequence):
            new_seq = row
        else:
            new_seq = []
            for i, key in enumerate(self.stat_params):
                if key in self.stat_params:
                    new_seq.append(self.stat_params[key])
                else:
                    new_seq.append(self.placeholder)

        if not header:
            if self.size_type == "Kb":
                for i in self._size_indexes:
                    new_seq[i] = new_seq[i] / 1024
            new_seq = self._round_data(new_seq)
            self.data.append(new_seq)
        csv_file.writerow(new_seq)
        base_file.flush()

    def _get_file_abstractions(self, filename: str) -> Sequence:
        """Получение необходимых абстракций файлов."""

        dummy_file = open(filename, mode='w', encoding="utf-8", newline='')
        dummy_csv = csv.writer(dummy_file)
        return dummy_file, dummy_csv

    # Метрики (перенесено и модифицировано из metrics/metrics.py от MaksFuji для удобства)

    @staticmethod
    def mse_metric(image1: np.ndarray, image2: np.ndarray) -> float:
        """Расчёт метрики MSE.
        На вход подаются две картинки в формате cv2 (numpy)."""

        mse = np.mean((image1 - image2) ** 2)

        return mse

    @staticmethod
    def ssim_metric(image1: np.ndarray, image2: np.ndarray) -> float:
        """Расчёт метрики SSIM.
        На вход подаются две картинки в формате cv2 (numpy)."""

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

        return score

    @classmethod
    def psnr_metric(cls, image1: np.ndarray, image2: np.ndarray) -> float:
        """Расчёт метрики PSNR.
        На вход подаются две картинки в формате cv2 (numpy)."""

        mse = cls.mse_metric(image1, image2)
        if mse == 0:
            return 100

        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

        return psnr

    # Интерфейс

    def change_file(self, new_filename: str = "statistics") -> bool:
        """Изменение файла."""

        try:
            dummy_file, dummy_csv = self._get_file_abstractions(new_filename)

            self.filename = new_filename
            self._file = dummy_file
            self._csv = dummy_csv

            self.write_stat(self.stat_params, header=True)
            return True
        except:
            return False

    def write_stat(self, row: Union[Sequence, Mapping], header=False) -> None:
        """Запись в нужный файл с выталкиванием буфера."""

        self._write_in_stat_file(self._csv, self._file, row, header=header)

    def write_summary(self, conf_p=0.95, summary_filename: str = "") -> None:
        """Запись итоговых результатов со статистической обработкой."""

        if not summary_filename:
            summary_filename = self.summary_definer.join(os.path.splitext(self.filename))
        summary_file, summary_csv = self._get_file_abstractions(summary_filename)
        self._write_in_stat_file(summary_csv, summary_file, self.summary_params, header=True)

        data_frame = pd.DataFrame.from_records(data=self.data, columns=self.stat_params)
        for i, value in enumerate(data_frame):
            if self.nominal_types[i] is str:
                continue

            row = data_frame[value]
            count = row.count()
            mean = row.mean()
            median = row.median()
            min_ = row.min()
            max_ = row.max()
            var = row.var()
            std = row.std()
            student = scipy.stats.t.ppf((1 + conf_p) / 2, count - 1)
            conf = student * std / (count ** 0.5)
            conf_min = mean - conf
            conf_max = mean + conf
            if self.rounder:
                mean = round(mean, self.rounder)
                median = round(median, self.rounder)
                var = round(var, self.rounder)
                std = round(std, self.rounder)
                conf = round(conf, self.rounder)
                conf_min = round(conf_min, self.rounder)
                conf_max = round(conf_max, self.rounder)

            new_summary_value = [value, count,
                                 mean, median,
                                 min_, max_,
                                 var, std,
                                 conf_p, conf, conf_min, conf_max]
            self._write_in_stat_file(summary_csv, summary_file, new_summary_value, header=True)

        total_count = data_frame["id"].count()
        nan_count = data_frame["ssim"].isna().sum()
        black_frame_rate = nan_count/total_count
        black_frame_rate_value = [
            "black_frame_rate", nan_count,
            black_frame_rate, black_frame_rate,
            black_frame_rate, black_frame_rate,
            0, 0,
            0, black_frame_rate, black_frame_rate
        ]
        self._write_in_stat_file(summary_csv, summary_file, black_frame_rate_value, header=True)
        summary_file.flush()
        summary_file.close()

    # Прочий технический метод.

    def cleanup(self):
        self.write_summary()
        self._file.close()
