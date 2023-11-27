import os
import queue
import socket
import threading
import time

import cv2

from common.logging_sd import configure_logger
from compress import createSd
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save, USE_PREDICTION, Platform, WARM_UP, \
    WINDOW_NAME, QUEUE_MAXSIZE_CLIENT_SD, NDARRAY_SHAPE_AFTER_SD
from core import latent_to_img
from utils import save_img, create_dir

logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_CLIENT_SD)


def uncompress(img):
    decoderStartTime = time.time()
    res = latent_to_img(img)
    logger.debug(f"time for clear decoder: {time.time()-decoderStartTime}")
    return res


def worker():
    global queue_of_frames

    createSd(Platform.CLIENT)
    count = 0
    is_warmup = True

    if USE_PREDICTION:
        sock_for_prediction = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_for_prediction.connect(('localhost', 9091))
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        try:
            if queue_of_frames.qsize() == 0:
                pass
            else:
                logger.debug(f"queue is not empty. Waiting for frame № {count} to decode")

                compressed_img = queue_of_frames.get()

                result_img = uncompress(compressed_img)

                if is_warmup:
                    # warm = result_img.tobytes()
                    is_warmup = False
                else:
                    dir_name = count
                    if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
                        create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
                    save_parent_dir_name = f"{dir_name}_run"

                    if is_save:
                        save_img(result_img, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

                    logger.debug(f"Display/send {count} frame")
                    # logger.debug(f"Shape is {result_img.shape}")
                    # logger.debug(len(result_img.tobytes()))

                    if USE_PREDICTION:
                        sock_for_prediction.sendall(result_img.tobytes())
                    else:
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

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 9090))
    sock.listen(1)
    con, _ = sock.accept()  # принимаем клиента

    print('Sock name: {}'.format(sock.getsockname()))

    img_size_to_receive = 1
    for dem in NDARRAY_SHAPE_AFTER_SD:
        img_size_to_receive *= dem

    # if not USE_PREDICTION:
    #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:

        received_bytes = b''
        while len(received_bytes) < img_size_to_receive:
            # logger.debug(f"len of r_b = {len(received_bytes)}")
            chunk = con.recv(img_size_to_receive - len(received_bytes))
            if not chunk:
                break
            received_bytes += chunk

        logger.debug(f"Got new frame, it's len is {len(received_bytes)}")

        # compress_img = con.recv(30000)  # получаем данные от клиента

        if queue_of_frames.qsize() >= QUEUE_MAXSIZE_CLIENT_SD:
            queue_of_frames.get_nowait()
        queue_of_frames.put(received_bytes)

    queue_of_frames.join()
    con.close()  # закрываем подключение


if __name__ == '__main__':
    main()
