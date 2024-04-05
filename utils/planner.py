import importlib
import configparser
import os
import sys
import copy
import json
import time


class ExperimentPlanner:
    """Планировщик экспериментов.
    Нужно определять в planner_sections.ini варианты параметров для перебора, разделяемые |.
    Перебираются все варианты и пишутся в директорию statistics.
    index.txt, 1.csv, 2.csv, ...
    Идёт подстановка в experiment_config.ini, затем возврат."""

    splitter = "|"
    extension = "csv"

    def __init__(self, sections_path: str,
                 experiment_config_path: str,
                 experiment_module: str,
                 statistics_path: str = "statistics",
                 index_filename: str = "index.txt"):
        """planner_config_path -- файл конфигурации планировщика.
        experiment_config_path -- файл конфигурации эксперимента.
        experiment_module -- модуль (скрипт) экспериментального пайплайна.
        statistics_path -- куда писать статистику."""

        self._experiment_module = experiment_module
        self.original_config = ""
        self.default_config = configparser.ConfigParser()
        self.experiment_config_path = experiment_config_path
        self.default_config.read(self.experiment_config_path, encoding="utf-8")
        with open(experiment_config_path, mode='r', encoding="utf-8") as orig:
            self.original_config = orig.read()
        self.planner_config = configparser.ConfigParser()
        self.planner_config.read(sections_path, encoding="utf-8")
        self.statistics_path = statistics_path
        self.index_filename = os.path.join(self.statistics_path, index_filename)

        self.default_config = self.config_to_dict(self.default_config, default=True)
        self.planner_config = self.config_to_dict(self.planner_config)

        os.makedirs(self.statistics_path, exist_ok=True)

    # Основной метод

    def run_experiment_series(self, since: int = 0):
        """Запустить серию экспериментов.
        since -- с которого испытания по счёту."""

        all_variants = self.get_all_combinations(self.planner_config)
        with open(self.index_filename, 'w', encoding="utf-8"):
            pass

        for i, variant in enumerate(all_variants):
            if i < since:
                continue

            save_filename = os.path.join(self.statistics_path, "{}.{}".format(i, self.extension))
            json_variant = json.dumps(variant, indent=4, ensure_ascii=False)

            if "ExperimentSettings" not in variant:
                variant["ExperimentSettings"] = dict()
            variant["ExperimentSettings"]["stat_filename"] = save_filename

            full_variant = self.inject_configuration(variant)
            config_variant = self.config_from_dict(full_variant)
            with open(self.experiment_config_path, 'w', encoding="utf-8") as efile:
                config_variant.write(efile, space_around_delimiters=False)

            checkpoint = time.time()
            print("===<>=== Испытание {} ===<>===".format(i))
            if self._experiment_module in sys.modules:
                importlib.reload(sys.modules[self._experiment_module])
            else:
                importlib.import_module(self._experiment_module)
            print("> Испытание {} завершено за {:.2f} с".format(i, time.time() - checkpoint))
            print()

            with open(self.index_filename, 'a', encoding="utf-8") as ifile:
                print("========= {} =========".format(save_filename), file=ifile)
                print(json_variant, file=ifile)
                print(file=ifile)

        with open(self.experiment_config_path, 'w', encoding="utf-8") as efile:
            efile.write(self.original_config)

    # Технические методы

    def get_all_combinations(self, dicter: dict, exclude_sections: list = None) -> list[dict]:
        """Получить все элементарные комбинации испытаний.
        dicter -- словарь вариантов конфигурации."""

        config_variants = []
        true_max_len = 1
        if not exclude_sections:
            exclude_sections = []
        delegated = False

        for section in dicter:
            if (section in exclude_sections) or (not dicter[section]):
                continue
            else:
                exclude_sections.append(section)
            sec_keys = list(dicter[section].keys())

            check_val = dicter[section][sec_keys[0]]
            if not isinstance(check_val, list):
                max_len = 1
            else:
                max_len = len(check_val)

            if max_len == 1:
                continue
            elif max_len > true_max_len:
                true_max_len = max_len

            for i in range(max_len):
                temp_dict = copy.deepcopy(dicter)

                for key, value in dicter[section].items():
                    temp_dict[section][key] = value[i]

                exclude_sec = exclude_sections.copy()
                new_config_variants = self.get_all_combinations(temp_dict, exclude_sections=exclude_sec)
                config_variants.extend(new_config_variants)
                delegated = True

            if delegated:
                break

        if true_max_len == 1:
            config_variants.append(dicter)

        return config_variants

    def inject_configuration(self, dicter: dict) -> dict:
        """Выполнить инъекцию варинатов конфигурации.
        dicter -- словарь с частичной конфигурацией."""

        new_dict = copy.deepcopy(self.default_config)
        for section in dicter:
            for key, value in dicter[section].items():
                new_dict[section][key] = dicter[section][key]

        return new_dict

    def config_to_dict(self, config: configparser.ConfigParser, default=False) -> dict:
        """Преобразование конфигурации в словарь.
        config -- конфигурация.
        default -- является ли конфигурация по умолчанию."""

        dicter = dict()
        max_lenner = dict()
        for section in config.sections():
            dicter[section] = dict(config[section])
            max_lenner[section] = 1

            for key, value in dicter[section].items():
                if self.splitter in value:
                    new_value = value.split(self.splitter)
                    dicter[section][key] = new_value

                    if len(new_value) > max_lenner[section]:
                        max_lenner[section] = len(new_value)

        if not default:
            for section in dicter:
                if max_lenner[section] > 1:
                    for key, value in dicter[section].items():
                        some_bad = False
                        if not isinstance(value, list):
                            value = [value]
                            some_bad = True
                        if len(value) < max_lenner[section]:
                            len_diff = max_lenner[section] - len(value)
                            value = value + [self.default_config[section][key]] * len_diff
                            some_bad = True
                        if some_bad:
                            dicter[section][key] = value

        return dicter

    def config_from_dict(self, dicter: dict) -> configparser.ConfigParser:
        """Преобразование словаря в конфигурацию."""

        for section in dicter:
            for key, value in dicter[section].items():
                if isinstance(value, list):
                    new_value = self.splitter.join(value)
                    dicter[section][key] = new_value

        config = configparser.ConfigParser()
        config.read_dict(dicter)

        return config
