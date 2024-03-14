import logging
import sys
import os

from deprecated.constants.constant import DEBUG, DATA_LOGS


def configure_logger(name):
    if DEBUG:
        module_name = name.strip('_')
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
                "%(name)s | %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s |"
                " %(message)s")

        if not os.path.exists(DATA_LOGS):
            os.makedirs(DATA_LOGS, exist_ok=True)

        file_handler = logging.FileHandler(
                f"{DATA_LOGS}/{module_name}.log", mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        module_name = name.strip('_')
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
                "%(name)s | %(asctime)s | %(levelname)s | %(message)s")

        if not os.path.exists(DATA_LOGS):
            os.makedirs(DATA_LOGS, exist_ok=True)

        file_handler = logging.FileHandler(
                f"{DATA_LOGS}/{module_name}.log", mode='w')
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
