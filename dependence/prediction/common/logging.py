import logging
import sys
import os


def configure_logger(name):
    logs_dir = 'logs/'
    module_name = name.strip('_')
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(name)s | %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s |"
        " %(message)s"
        )

    os.makedirs(logs_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        f"{logs_dir}{module_name}.log", mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
