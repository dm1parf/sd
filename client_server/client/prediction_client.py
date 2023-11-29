import os
import queue
import socket
import socketserver
import threading
import time

import cv2
import numpy as np

from client_server.core import connection_utill
from common.logging_sd import configure_logger
from constants.constant import PREDICTION_MODEL_PATH, WINDOW_NAME, QUEUE_MAXSIZE_CLIENT_PREDICTION, \
    MAXSIZE_OF_RESTORED_IMGS_LIST, NUMBER_OF_FRAMES_TO_PREDICT, NDARRAY_SHAPE_AFTER_SD, DEVICE, \
    USE_OPTIMIZED_PREDICTION, OPTIMIZED_PREDICTION_MODEL_PATH, VIDEO_CLIENT_URL, \
    VIDEO_CLIENT_PORT, SEND_VIDEO, SHOW_VIDEO, PREDICTION_CLIENT_URL, PREDICTION_CLIENT_PORT
from prediction import Model, DMVFN
from prediction.model.models import DMVFN_optim

logger = configure_logger(__name__)
queue_of_frames = queue.Queue(QUEUE_MAXSIZE_CLIENT_PREDICTION)
img_size_to_receive = 1


class PredictionServerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global queue_of_frames
        global img_size_to_receive

        received_bytes = b''
        while len(received_bytes) < img_size_to_receive:
            chunk = self.request.recv(img_size_to_receive - len(received_bytes))
            if not chunk:
                break
            received_bytes += chunk

        logger.debug(f"Got new frame, it's len is {len(received_bytes)}")

        compress_img = np.frombuffer(received_bytes, dtype=np.uint8).reshape(NDARRAY_SHAPE_AFTER_SD)

        if queue_of_frames.qsize() >= QUEUE_MAXSIZE_CLIENT_PREDICTION:
            queue_of_frames.get_nowait()
        queue_of_frames.put(compress_img)


def predict_img(list_of_images, model):
    predictionStartTime = time.time()
    res = model.predict(list_of_images[-MAXSIZE_OF_RESTORED_IMGS_LIST:])
    logger.debug(f"time for clear prediction: {time.time() - predictionStartTime}")
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

    if USE_OPTIMIZED_PREDICTION:
        prediction_model = Model(
            DMVFN_optim(os.path.abspath(f"../../" + PREDICTION_MODEL_PATH),
                        os.path.abspath(f"../../" + OPTIMIZED_PREDICTION_MODEL_PATH),
                        DEVICE
                        ))
    else:
        prediction_model = Model(
            DMVFN(os.path.abspath(f"../../" + PREDICTION_MODEL_PATH), DEVICE))

    restored_imgs = []
    is_first_frame = True
    number_of_frame = 0

    if SHOW_VIDEO and not SEND_VIDEO:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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

                if SEND_VIDEO:
                    connection_utill.send_message(VIDEO_CLIENT_URL, VIDEO_CLIENT_PORT, result_img.tobytes())
                elif SHOW_VIDEO:
                    cv2.imshow(WINDOW_NAME, result_img)
                    cv2.waitKey(25)

            number_of_frame += 1
            queue_of_frames.task_done()


def main():
    global queue_of_frames
    global img_size_to_receive

    for dem in NDARRAY_SHAPE_AFTER_SD:
        img_size_to_receive *= dem

    threading.Thread(target=worker, daemon=True).start()

    connection_utill.create_server(PREDICTION_CLIENT_URL, PREDICTION_CLIENT_PORT, PredictionServerHandler)


if __name__ == '__main__':
    main()
