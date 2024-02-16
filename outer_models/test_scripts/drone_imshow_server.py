"""
Среду устанавливать достаточно хлопотно, но примерно так:

apt uninstall python3
apt install libgstrtspserver-1.0-dev
apt install gstreamer1.0-rtsp
apt install libgirepository1.0-dev
apt install gir1.2-gtk-3.0
apt install gcc
apt install libcairo2-dev
apt install pkg-config
apt install python3.11-dev
apt install libglib2.0-dev
python3.11 -m pip install pycairo
python3.11 -m pip install PyGObject
"""


import signal
import struct
import time
import threading
import socket
import configparser
import numpy as np
import cv2


class DroneImshowServer:
    def __init__(self, width: int, height: int, fps: int, internal_ip: str, internal_port: int,
                 sync_mode: bool = False, window_title: str = "==="):
        """video -- файл видео."""

        super().__init__()

        self._new_socket = 0
        self._exit_thread = False
        self._socket_data = (internal_ip, internal_port)
        self._this_frame = None
        self._frame_lock = threading.Lock()
        self._frame_thread = threading.Thread(target=self.get_frame_from_socket, daemon=True)

        self.width = width
        self.height = height
        self.frame_num = 0
        self.fps = fps
        self.sec_duration = 1 / self.fps
        self.duration = int(1000 * self.sec_duration)
        self.window_title = window_title
        self.sync_mode = sync_mode

        signal.signal(signal.SIGINT, self.close_internal_socket)

        self._frame_thread.start()

    def close_internal_socket(self, *_, **__):
        """Закрыть сокет."""

        self._exit_thread = True
        # self._new_socket.shutdown()
        self._new_socket.close()

    def image_show_loop(self):
        """Запись кадра в буфер при запросе."""

        print("=== Imshow-сервер запущен! ===")
        while True:
            if (self._this_frame is not None) and (not self.sync_mode):
                cv2.imshow(self.window_title, self._this_frame)
                cv2.waitKey(self.duration)
            else:
                time.sleep(self.sec_duration)

    def get_frame_from_socket(self):
        """Получение кадров из сокета."""

        self._new_socket = socket.socket()
        self._new_socket.bind(self._socket_data)
        self._new_socket.listen(1)
        connection, address = self._new_socket.accept()
        while True:
            if self._exit_thread:
                break
            try:
                image: bytes = connection.recv(4)
                if not image:
                    connection, address = self._new_socket.accept()
                    continue

                image_len = struct.unpack('I', image)[0]
                image: bytes = connection.recv(image_len)
                while len(image) != image_len:
                    diff = image_len - len(image)
                    image += connection.recv(diff)

                self._frame_lock.acquire()
                self._this_frame = np.frombuffer(image, dtype=np.uint8)
                self._this_frame = self._this_frame.reshape([self.height, self.width, 3])
                if self.sync_mode:
                    cv2.imshow(self.window_title, self._this_frame)
                    cv2.waitKey(1)
                self._frame_lock.release()

            except (ConnectionResetError, socket.error):
                if self._exit_thread:
                    break
                connection, address = self._new_socket.accept()
                continue


if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('./outer_models/test_scripts/drone_imshow_server.ini', encoding='utf-8')
    video_settings = parser["VideoSettings"]
    internal_stream_settings = parser["InternalStreamSettings"]

    width = int(video_settings["width"])
    height = int(video_settings["height"])
    fps = int(video_settings["fps"])
    window_title = video_settings["window_title"]
    sync_mode = bool(int(video_settings["sync_mode"]))

    internal_ip = internal_stream_settings["host"]
    internal_port = int(internal_stream_settings["port"])

    server = DroneImshowServer(width, height, fps, internal_ip, internal_port, sync_mode, window_title)
    server.image_show_loop()
