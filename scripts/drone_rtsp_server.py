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
import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst, GstRtspServer, GLib, GstRtsp


class DroneRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width: int, height: int, fps: int, internal_ip: str, internal_port: int):
        """video -- файл видео."""

        super().__init__()

        self._new_socket = 0
        self._exit_thread = False
        self._socket_data = (internal_ip, internal_port)
        self._this_frame = b''
        self._frame_lock = threading.Lock()
        self._frame_thread = threading.Thread(target=self.get_frame_from_socket, daemon=True)

        self.width = width
        self.height = height
        self.frame_num = 0
        self.fps = fps
        self.duration = 1 / self.fps * Gst.SECOND

        self.pipe = ("appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
                     "caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
                     "! videoconvert ! video/x-raw,format=I420 "
                     "! jpegenc "
                     "! queue ! rtpjpegpay name=pay0 pt=96").format(fps=self.fps, width=self.width, height=self.height)
        self.pipeline = Gst.parse_launch(self.pipe)
        self.loop = None
        self.appsrc=self.pipeline.get_by_name('source')
        self.appsrc.connect('need-data', self.write_to_buffer)

        self._frame_thread.start()

    def close_internal_socket(self, *_, **__):
        """Закрыть сокет."""

        self._exit_thread = True
        self._new_socket.shutdown()
        self._new_socket.close()

    def do_create_element(self, url):
        """Вещание видео для подключившегося."""

        return self.pipeline

    def write_to_buffer(self, src, lenght):
        """Запись кадра в буфер при запросе."""

        self._frame_lock.acquire()
        data = self._this_frame
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        self._frame_lock.release()

        pts = self.frame_num * self.duration
        buf.dts = pts
        buf.pts = pts
        buf.offset = pts
        self.frame_num += 1
        src.emit('push-buffer', buf)

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
                self._this_frame = image
                self._frame_lock.release()

            except (ConnectionResetError, socket.error):
                if self._exit_thread:
                    break
                connection, address = self._new_socket.accept()
                continue


class DroneRTSPServerManager:
    def __init__(self, port: str, streamer: str, auth: list,
                 width: int, height: int, fps: int,
                 internal_ip: str, internal_port: int,
                 verbose: bool = False):
        self.rtsp_server = GstRtspServer.RTSPServer()
        self.rtsp_server.set_service(port)
        self.port = port
        self.auth = auth
        self.streamer = streamer
        self.verbose = verbose
        Gst.init(None)
        self._factory = DroneRTSPMediaFactory(width, height, fps, internal_ip, internal_port)

        self.loop = GLib.MainLoop()
        signal.signal(signal.SIGINT, self.close_loop)

        if auth:
            self._rtsp_auth = GstRtspServer.RTSPAuth()
            self._auth_token = GstRtspServer.RTSPToken()
            self._auth_token.set_string("media.factory.role", auth[0])
            self._basic_auth = GstRtspServer.RTSPAuth.make_basic(*auth)
            self._rtsp_auth.add_basic(self._basic_auth, self._auth_token)
            self._rtsp_auth.add_digest(*auth, self._auth_token)

            self.rtsp_server.set_auth(self._rtsp_auth)
            self._permissions = GstRtspServer.RTSPPermissions()
            self._permissions.add_permission_for_role(auth[0], "media.factory.access", True)
            self._permissions.add_permission_for_role(auth[0], "media.factory.construct", True)
            self._factory.set_permissions(self._permissions)

        self._factory.set_shared(True)
        self._mount_points = self.rtsp_server.get_mount_points()
        self._mount_points.add_factory("/{}".format(self.streamer), self._factory)
        self.rtsp_server.attach(None)

    def close_loop(self, *_):
        if self.verbose:
            print("\n=== Выключаем RTSP-стрим с БПЛА...")
        self._factory.close_internal_socket()
        self.loop.quit()
        exit()

    def start_loop(self):
        if self.verbose:
            print("=== Запускаем RTSP-стрим с БПЛА...\n")
        if auth:
            auth_str = ":".join(auth)
        else:
            auth_str = ""
        if self.verbose:
            print("RTSP-вещание с БПЛА доступно по URI: rtsp://{}@(IP_адрес):{}/{}".format(auth_str,
                                                                                    self.port, self.streamer))

        self.loop.run()


if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('./dependence/materials/drone_rtsp_server.ini')
    rtsp_settings = parser["RtspSettings"]
    video_settings = parser["VideoSettings"]
    internal_stream_settings = parser["InternalStreamSettings"]

    port = rtsp_settings["port"]
    stream_name = rtsp_settings["stream_name"]
    auth = [rtsp_settings["user"], rtsp_settings["password"]]
    verbose = bool(int(rtsp_settings["verbose"]))

    width = int(video_settings["width"])
    height = int(video_settings["height"])
    fps = int(video_settings["fps"])

    internal_ip = internal_stream_settings["host"]
    internal_port = int(internal_stream_settings["port"])

    server = DroneRTSPServerManager(port, stream_name, auth, width, height, fps,
                                    internal_ip, internal_port, verbose)
    server.start_loop()
