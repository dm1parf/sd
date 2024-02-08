import os
import csv
from typing import Union, Sequence, Mapping
import scipy
import pandas as pd


class StatWriter:
    """Класс для записи статистических данных в csv."""

    summary_definer = "summary"
    nominal_types = [int, str,
                     float, float, float,
                     int, int,
                     float, float,
                     float, float,
                     float, float,
                     float, float, float]
    stat_params = ["id", "name",
                   "psnr", "mse", "ssim",
                   "latent_size", "min_size",
                   "encoding_time", "decoding_time",
                   "cypher_time", "uncypher_time",
                   "superresolution_time", "predictor_time",
                   "total_coder_time", "total_decoder_time", "total_time"]
    # id -- идентификатор эксперимента.
    # name -- имя файла.
    # psnr -- метрика PSNR.
    # mse -- метрика MSE.
    # ssim -- метрика SSIM.
    # latent_size -- размер латентного пространства до сжатия.
    # min_size -- минимальный размер в сжатом виде.
    # encoding_time -- время кодирования.
    # decoding_time -- время декодирования.
    # cypher_time -- время шифрования.
    # uncypher_time -- время дешифрования.
    # superresolution_time -- время суперрезолюции (на одну картинку).
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

    def __init__(self, filename: str = "statistics.csv", placeholder="-"):
        self.placeholder = placeholder
        self.filename = ""
        self._file = None
        self._csv = None
        self.data = []  # Явно сохраняем данные для дальнейшего анализа.

        self.change_file(filename)

    # Некоторые технические методы

    def _write_in_stat_file(self, csv_file, base_file, row: Union[Sequence, Mapping]) -> None:
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
        csv_file.writerow(new_seq)
        base_file.flush()

    def _get_file_abstractions(self, filename: str) -> Sequence:
        """Получение необходимых абстракций файлов."""

        dummy_file = open(filename, encoding="utf-8", newline='')
        dummy_csv = csv.writer(dummy_file)
        return dummy_csv, dummy_csv

    # Интерфейс

    def change_file(self, new_filename: str = "statistics") -> bool:
        """Изменение файла."""

        try:
            dummy_file, dummy_csv = self._get_file_abstractions(new_filename)

            self.filename = new_filename
            self._file = dummy_file
            self._csv = dummy_csv

            self.write_stat(self.stat_params)
            return True
        except:
            return False

    def write_stat(self, row: Union[Sequence, Mapping]) -> None:
        """Запись в нужный файл с выталкиванием буфера."""

        self._write_in_stat_file(self._csv, self._file, row)

    def write_summary(self, interval=0.95, summary_filename: str = "") -> None:
        """Запись итоговых результатов со статистической обработкой."""

        if not summary_filename:
            summary_filename = self.summary_definer.join(os.path.splitext(self.filename))
        summary_file, summary_csv = self._get_file_abstractions(summary_filename)
        self._write_in_stat_file(summary_csv, summary_file, self.summary_params)

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
            self._write_in_stat_file(summary_csv, summary_file, new_summary_value)

        summary_file.flush()
        summary_file.close()

    # Прочий технический метод.

    def __del__(self):
        self.write_summary()
        self._file.close()
