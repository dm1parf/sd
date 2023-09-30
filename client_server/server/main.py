import concurrent.futures
import os
import threading
import time
import socket
from multiprocessing import Queue

from compress import run_coder, run_decoder, createSd
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_quantize, save_rescaled_out, Platform
from core import load_and_rescaled
from common.logging_sd import configure_logger
import cv2


def compress(img):
    start = time.time()
    res = run_coder(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    logger.debug(f"Time for clear coder: {time.time() - start}")
    return res


def worker():
    while True:
        item = queue_of_futures.get()
        startTime = time.time()
        while item.running():
            pass
        logger.debug(f"Time for sending is {time.time() - startTime}")
        sock.sendall(item.result())
        data = sock.recv(1024)  # получаем данные с сервера
        print("Server sent: ", data.decode())
        queue_of_futures.task_done()


createSd(Platform.SERVER)

logger = configure_logger(__name__)

logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

if not os.path.exists(DIR_PATH_INPUT):
    os.makedirs(DIR_PATH_INPUT)
if not os.path.exists(DIR_PATH_OUTPUT):
    os.makedirs(DIR_PATH_OUTPUT)

logger.debug(f"get files in dir = {DIR_NAME}")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 9090))

queue_of_futures = Queue()
with concurrent.futures.ThreadPoolExecutor() as executor:
    threading.Thread(target=worker, daemon=True).start()
    while True:
        for rescaled_img, image, img_name, save_parent_dir_name, save_dir_name in load_and_rescaled():
            queue_of_futures.put(executor.submit(compress, rescaled_img))

    queue_of_futures.join()
    print('Close')
    sock.close()
