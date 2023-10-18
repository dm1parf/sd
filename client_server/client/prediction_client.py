import os
import queue
import socket
import threading
import time
import numpy as np

import cv2

from common.logging_sd import configure_logger
from compress import createSd
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save, PREDICTION_MODEL_PATH, REAL, FAKE, REAL_NAME, \
    FAKE_NAME, USE_PREDICTION, Platform, QUEUE_MAXSIZE_CLIENT, WARM_UP, WINDOW_NAME, QUEUE_MAXSIZE_CLIENT_PREDICTION, \
    MAXSIZE_OF_RESTORED_IMGS_LIST, NUMBER_OF_FRAMES_TO_PREDICT, NDARRAY_SHAPE_AFTER_SD
from core import latent_to_img
from prediction import Model, DMVFN
from utils import save_img, create_dir


logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_CLIENT_PREDICTION)


def predict_img(list_of_images, model):
    predictionStartTime = time.time()
    res = model.predict(list_of_images[-MAXSIZE_OF_RESTORED_IMGS_LIST:])
    logger.debug(f"time for clear prediction: {time.time()-predictionStartTime}")
    return res


def add_frame_to_list(list_of_frames, img_to_add):
    list_of_frames.append(img_to_add)

    if len(list_of_frames) > MAXSIZE_OF_RESTORED_IMGS_LIST:
        del list_of_frames[0]

    return list_of_frames


def get_frame_from_future(list_of_imgs, number_of_frames_to_predict, prediction_model):
    if number_of_frames_to_predict <= 0:
        return list_of_imgs
    else:
        number_of_frames_to_predict -= 1
        result_img = predict_img(list_of_imgs, prediction_model)

        list_of_imgs = add_frame_to_list(list_of_imgs, result_img)

        return get_frame_from_future(list_of_imgs, number_of_frames_to_predict, prediction_model)


def worker():
    global queue_of_frames

    prediction_model = Model(DMVFN(PREDICTION_MODEL_PATH, device="cpu"))
    restored_imgs = []
    is_first_frame = True
    number_of_frame = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # logger.debug("Starting warm up")
    # warm_up_start_time = time.time()
    # predict_img([WARM_UP_PREDICTION, WARM_UP_PREDICTION], prediction_model)
    # logger.debug(f"Model warmed up. Time for warm up: {time.time() - warm_up_start_time}")

    while True:
        if queue_of_frames.qsize() == 0:
            pass
        else:
            sd_img = queue_of_frames.get()

            restored_imgs = add_frame_to_list(restored_imgs, sd_img)

            logger.debug(f"len of arr is {len(restored_imgs)} for {number_of_frame} frame")

            if not len(restored_imgs) < MAXSIZE_OF_RESTORED_IMGS_LIST:

                result_img = predict_img(restored_imgs, prediction_model)

                restored_imgs = add_frame_to_list(restored_imgs, result_img)

                if is_first_frame:
                    restored_imgs = get_frame_from_future(restored_imgs, NUMBER_OF_FRAMES_TO_PREDICT, prediction_model)
                    result_img = restored_imgs[-1]
                    is_first_frame = False

                cv2.imshow(WINDOW_NAME, result_img)
                cv2.waitKey(25)

            number_of_frame += 1
            queue_of_frames.task_done()


def main():
    global queue_of_frames

    img_size_to_receive = 1
    for dem in NDARRAY_SHAPE_AFTER_SD:
        img_size_to_receive *= dem

    threading.Thread(target=worker, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 9091))
    sock.listen(1)
    con, _ = sock.accept()  # принимаем клиента

    print('Sock name: {}'.format(sock.getsockname()))

    while True:

        received_bytes = b''
        while len(received_bytes) < img_size_to_receive:
            # logger.debug(f"len of r_b = {len(received_bytes)}")
            chunk = con.recv(img_size_to_receive - len(received_bytes))
            if not chunk:
                break
            received_bytes += chunk

        logger.debug(f"Got new frame, it's len is {len(received_bytes)}")

        compress_img = np.frombuffer(received_bytes, dtype=np.uint8).reshape(NDARRAY_SHAPE_AFTER_SD)

        if queue_of_frames.qsize() >= QUEUE_MAXSIZE_CLIENT_PREDICTION:
            queue_of_frames.get_nowait()
        queue_of_frames.put(compress_img)

    queue_of_frames.join()
    con.close()  # закрываем подключение


if __name__ == '__main__':
    main()
