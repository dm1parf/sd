import queue
import socket
import threading

import cv2
import numpy as np

from client_server.core import connection_utill
from common.logging_sd import configure_logger
from constants.constant import WINDOW_NAME, NDARRAY_SHAPE_AFTER_SD
from utils import load_frame_video

logger = configure_logger(__name__)
MAX_FRAME = 1000
queue_of_frames = queue.Queue(MAX_FRAME)


def req(con, img_size_to_receive):
    received_bytes = b''
    while len(received_bytes) < img_size_to_receive:
        # logger.debug(f"len of r_b = {len(received_bytes)}")
        chunk = con.recv(img_size_to_receive - len(received_bytes))
        if not chunk:
            break
        received_bytes += chunk

    logger.debug(f"Got new frame, it's len is {len(received_bytes)}")

    compress_img = np.frombuffer(received_bytes, dtype=np.uint8).reshape(NDARRAY_SHAPE_AFTER_SD)

    if queue_of_frames.qsize() >= MAX_FRAME:
        queue_of_frames.get_nowait()
    queue_of_frames.put(compress_img)


def worker():
    img_size_to_receive = 1
    for dem in NDARRAY_SHAPE_AFTER_SD:
        img_size_to_receive *= dem

    connection_utill.create_client('', 9092, req, img_size_to_receive)

    queue_of_frames.join()


def thread():
    threading.Thread(target=worker, daemon=True).start()


def main():
    thread()
    cv2.startWindowThread()
    cv2.namedWindow(WINDOW_NAME)

    while True:
        if queue_of_frames.qsize() == 0:
            pass
        else:
            cv2.imshow(WINDOW_NAME, queue_of_frames.get())
            cv2.waitKey(1)  # i love mac os... it is fix a non-displaying window on the maс os
            # logger.debug(f"new frame in video {video_name}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
