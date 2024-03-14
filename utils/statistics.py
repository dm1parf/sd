import os
import csv
import math
from typing import Union, Sequence, Mapping
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
                     int, int,
                     float, float,
                     float, float,
                     float, float,
                     float, float,
                     float, float, float]
    stat_params = ["id", "name",
                   "psnr", "mse", "ssim",
                   "latent_size", "min_size",
                   "encoding_time", "decoding_time",
                   "quant_time", "dequant_time",
                   "compress_time", "decompress_time",
                   "superresolution_time", "predictor_time",
                   "total_coder_time", "total_decoder_time", "total_time"]
    # id -- идентификатор испытания.
    # name -- имя файла.
    # psnr -- метрика PSNR.
    # mse -- метрика MSE.
    # ssim -- метрика SSIM.
    # latent_size -- размер латентного пространства до сжатия.
    # min_size -- минимальный размер в сжатом виде.
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

    summary_params = ["value", "count",
                      "mean", "med",
                      "abs_min", "abs_max",
                      "var", "std",
                      "conf", "conf_min", "conf_max"]
    # value -- величина.
    # count -- всего значений.
    # mean -- среднее.
    # med -- медиана.
    # abs_min -- абсолютный минимум.
    # abs_max -- абсолютный максимум.
    # var -- дисперсия.
    # std -- среднее.
    # conf -- доверительный интервал.
    # conf_min -- минимум с учётом доверительного интервала.
    # conf_max -- максимум с учётом доверительного интервала.

    def __init__(self, filename: str = None, placeholder="-"):
        self.placeholder = placeholder
        self.filename = ""
        if not filename:
            filename = self.default_filename
        self._file = None
        self._csv = None
        self.data = []  # Явно сохраняем данные для дальнейшего анализа.

        self.change_file(filename)

    # Некоторые технические методы

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

        score = structural_similarity(image1, image2, data_range=image2.max() - image2.min())

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

    def write_summary(self, interval=0.95, summary_filename: str = "") -> None:
        """Запись итоговых результатов со статистической обработкой."""

        if not summary_filename:
            summary_filename = self.summary_definer.join(os.path.splitext(self.filename))
        summary_file, summary_csv = self._get_file_abstractions(summary_filename)
        self._write_in_stat_file(summary_csv, summary_file, self.summary_params, header=True)

        data_frame = pd.DataFrame.from_records(data=self.data, columns=self.stat_params)
        for i, value in enumerate(data_frame):
            if self.nominal_types[i] == str:
                continue

            row = data_frame[value]
            count = row.count()
            mean = row.mean()
            median = row.median()
            min_ = row.min()
            max_ = row.max()
            var = row.var()
            std = row.std()
            student = scipy.stats.t.ppf((1 + interval) / 2, count - 1)
            delta = student * std / (count ** 0.5)
            conf_min = mean - delta
            conf_max = mean + delta

            new_summary_value = [value, count,
                                 mean, median,
                                 min_, max_,
                                 var, std,
                                 interval, conf_min, conf_max]
            self._write_in_stat_file(summary_csv, summary_file, new_summary_value, header=True)

        summary_file.flush()
        summary_file.close()

    # Прочий технический метод.

    def cleanup(self):
        self.write_summary()
        self._file.close()
