import os
import queue
import socket
import threading
import time

import cv2

from client_server.core import connection_utill
from common.logging_sd import configure_logger
from compress import run_coder, createSd
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_quantize, Platform, \
    QUEUE_MAXSIZE_SERVER, SD_CLIENT_URL, SD_CLIENT_PORT
from core import load_and_rescaled

logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_SERVER)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def compress(img):
    logger.debug("Coder started")
    start = time.time()
    res = run_coder(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    logger.debug(f"Time for clear coder: {time.time() - start}")
    return res


def worker():
    global queue_of_frames, sock
    connection_utill

    createSd(Platform.SERVER)

    while True:
        frame = compress(queue_of_frames.get())

        connection_utill.send_message(SD_CLIENT_URL, SD_CLIENT_PORT, frame)

        queue_of_frames.task_done()


def main():
    global queue_of_frames, sock

    logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT):
        os.makedirs(DIR_PATH_OUTPUT)

    logger.debug(f"get files in dir = {DIR_NAME}")

    threading.Thread(target=worker, daemon=True).start()
    while True:
        for rescaled_img, image, img_name, save_parent_dir_name, save_dir_name in load_and_rescaled():
            if queue_of_frames.qsize() >= QUEUE_MAXSIZE_SERVER:
                queue_of_frames.get_nowait()
            queue_of_frames.put(rescaled_img)

            time.sleep(1.1)

    queue_of_frames.join()
    print('Close')
    sock.close()


if __name__ == '__main__':
    main()
