import queue
import socket
import socketserver
import threading

import cv2
import numpy as np

from client_server.core import connection_utill
from common.logging_sd import configure_logger
from constants.constant import WINDOW_NAME, NDARRAY_SHAPE_AFTER_SD, VIDEO_CLIENT_PORT, VIDEO_CLIENT_URL
from utils import load_frame_video

logger = configure_logger(__name__)
MAX_FRAME = 1000
queue_of_frames = queue.Queue(MAX_FRAME)
img_size_to_receive = 1


class VideoClientServerHandler(socketserver.BaseRequestHandler):
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
        queue_of_frames.put(compress_img)


def worker():
    global img_size_to_receive
    for dem in NDARRAY_SHAPE_AFTER_SD:
        img_size_to_receive *= dem

    connection_utill.create_server(VIDEO_CLIENT_URL, VIDEO_CLIENT_PORT, VideoClientServerHandler)

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
