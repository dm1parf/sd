import os
import queue
import socketserver
import threading
import time

import cv2

from client_server.core import connection_utill
from common.logging_sd import configure_logger
from compress import createSd
from constants.sd_warmup_data import WARM_UP
from core import latent_to_img, run_warmup
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save, USE_PREDICTION, Platform, \
    WINDOW_NAME, QUEUE_MAXSIZE_CLIENT_SD, SHOW_VIDEO, PREDICTION_CLIENT_URL, PREDICTION_CLIENT_PORT, SD_CLIENT_URL, \
    SD_CLIENT_PORT
from utils import save_img, create_dir

logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_CLIENT_SD)


class SdServerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global queue_of_frames
        compress_img = self.request.recv(30000)
        if queue_of_frames.qsize() >= QUEUE_MAXSIZE_CLIENT_SD:
            queue_of_frames.get_nowait()
        queue_of_frames.put(compress_img)


def uncompress(img):
    decoderStartTime = time.time()
    res = latent_to_img(img)
    logger.debug(f"time for clear decoder: {time.time() - decoderStartTime}")
    return res


def worker():
    global queue_of_frames

    createSd(Platform.CLIENT)
    count = 0
    is_warmup = True

    # if USE_PREDICTION:
    #     sock_for_prediction = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     sock_for_prediction.connect(('localhost', 9091))
    # else:
    #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if not USE_PREDICTION and SHOW_VIDEO:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        try:
            if queue_of_frames.qsize() == 0:
                pass
            else:
                logger.debug(f"queue is not empty. Waiting for frame № {count} to decode")

                compressed_img = queue_of_frames.get()

                if is_warmup:
                    run_warmup(compressed_img)
                    is_warmup = False
                else:
                    result_img = uncompress(compressed_img)

                    dir_name = count
                    if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
                        create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
                    save_parent_dir_name = f"{dir_name}_run"

                    if is_save:
                        save_img(result_img, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

                    logger.debug(f"Display/send {count} frame")

                    if USE_PREDICTION:
                        # sock_for_prediction.sendall(result_img.tobytes())
                        connection_utill.send_message(PREDICTION_CLIENT_URL, PREDICTION_CLIENT_PORT, result_img.tobytes())
                    elif SHOW_VIDEO:
                        cv2.imshow(WINDOW_NAME, result_img)
                        cv2.waitKey(25)

                    count += 1
                queue_of_frames.task_done()
        except Exception as err:
            logger.error(f"Error while processing {count} frame. Reason: {err}")
            count += 1


def main():
    global queue_of_frames

    if not os.path.exists(DIR_PATH_INPUT):
        os.makedirs(DIR_PATH_INPUT)
    if not os.path.exists(DIR_PATH_OUTPUT):
        os.makedirs(DIR_PATH_OUTPUT)

    threading.Thread(target=worker, daemon=True).start()

    logger.debug(f"Starting warm up")
    warm_up_start_time = time.time()
    queue_of_frames.put(WARM_UP)

    queue_of_frames.join()

    logger.debug(f"Models warmed up. Time for warm up: {time.time() - warm_up_start_time}")

    connection_utill.create_server(SD_CLIENT_URL, SD_CLIENT_PORT, SdServerHandler)


if __name__ == '__main__':
    main()
