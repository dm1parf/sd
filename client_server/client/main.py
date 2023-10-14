import os
import queue
import socket
import threading
import time

import cv2

from common.logging_sd import configure_logger
from compress import createSd
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save, PREDICTION_MODEL_PATH, REAL, FAKE, REAL_NAME, \
    FAKE_NAME, USE_PREDICTION, Platform, QUEUE_MAXSIZE_CLIENT, WARM_UP, WINDOW_NAME
from core import latent_to_img
from prediction import Model, DMVFN
from utils import save_img, create_dir

logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_CLIENT)


def uncompress(img):
    decoderStartTime = time.time()
    res = latent_to_img(img)
    logger.debug(f"time for clear decoder: {time.time()-decoderStartTime}")
    return res


def predict_img(list_of_images, model):
    predictionStartTime = time.time()
    res = model.predict(list_of_images[-2:])
    logger.debug(f"time for clear prediction: {time.time()-predictionStartTime}")
    return res


def worker():
    global queue_of_frames

    createSd(Platform.CLIENT)
    count = 0

    if USE_PREDICTION:
        prediction_model = Model(DMVFN(PREDICTION_MODEL_PATH))
        pattern = [REAL_NAME] * REAL + [FAKE_NAME] * FAKE
        pattern_counter = 0
        restored_imgs = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        try:
            if queue_of_frames.qsize() == 0:
                pass
            else:
                logger.debug(f"queue is not empty. Waiting for future № {count} to complete")

                compressed_img = queue_of_frames.get()

                if USE_PREDICTION:
                    if pattern[pattern_counter % len(pattern)] == REAL_NAME:
                        result_img = uncompress(compressed_img)

                    elif pattern[pattern_counter % len(pattern)] == FAKE_NAME:
                        result_img = predict_img(restored_imgs, prediction_model)

                    restored_imgs.append(result_img)

                    if len(restored_imgs) > 2:
                        del restored_imgs[0]

                    pattern_counter += 1

                else:
                    result_img = uncompress(compressed_img)

                dir_name = count
                if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
                    create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
                save_parent_dir_name = f"{dir_name}_run"

                if is_save:
                    save_img(result_img, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

                logger.debug(f"Display {count} frame")

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

    while True:

        compress_img = con.recv(30000)  # получаем данные от клиента

        if queue_of_frames.qsize() >= QUEUE_MAXSIZE_CLIENT:
            queue_of_frames.get_nowait()
        queue_of_frames.put(compress_img)

        # time.sleep(DELAY_BETWEEN_FRAMES)

    queue_of_frames.join()
    con.close()  # закрываем подключение


if __name__ == '__main__':
    main()
