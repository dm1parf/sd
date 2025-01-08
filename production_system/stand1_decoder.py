import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import signal
import socket
import cv2
import time
import csv
import traceback
import multiprocessing
from collections import deque
import numpy as np
import math
import argparse
import shutil
from skimage.metrics import structural_similarity
from datetime import datetime
from production_system.packet_manager import PacketManager
from production_system.production_guardian import ConfigurationGuardian


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(prog="Эмулятор декодера FPV CTVP",
                                        description="Сделано для испытаний канала.")
    arguments.add_argument('-c', '--cut', dest="cut", type=int, default=1300)
    arguments.add_argument('--record', dest="record", action='store_true', default=False)
    arguments.add_argument("--device", dest="device", type=int, default=-1,
                           help="Устройство")
    args = arguments.parse_args()
    partition = args.cut
    record = args.record
    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


class PacketAccounter:
    def __init__(self, actual_list, pending_length=10, actual_length=5):
        self._pending_dict = dict()
        self._pending_deque = deque(maxlen=pending_length)  # FIFO
        # self._actual_deque = deque(maxlen=actual_length)  # LIFO
        self._internal_actual_list = []
        self._actual_list = actual_list

        self.pending_length = pending_length
        self.actual_length = actual_length
        self._len_counter = 0

        self._bin_lens = []
        self._sec_interval = 1.0
        self._summ_bytes = 0

    def get_delta_sec(self, datetime_old, datetime_new):
        """Получить длину между записями."""

        delta = datetime_new - datetime_old
        total_sec = delta.total_seconds()

        return total_sec

    def get_sync_actual_array(self):
        """Получить синхронизированный список."""

        return self._actual_list

    def get_actual_msg(self):
        """Получить актуальное сообщение."""

        try:
            actual_msg = self._actual_list.pop()
            if actual_msg:
                return actual_msg
            else:
                return None
        except IndexError:
            return None

    def add_actual_msg(self, msg_data):
        """Добавить актуальное сообщение."""

        now_datetime = datetime.utcnow()
        payload_length = len(msg_data["payload"])
        lenner = [now_datetime, payload_length]
        self._summ_bytes += payload_length
        self._bin_lens.append(lenner)
        i = 0
        while i < len(self._bin_lens):
            first_interval = self.get_delta_sec(self._bin_lens[i][0], now_datetime)
            if first_interval > self._sec_interval:
                _, old_length = self._bin_lens.pop(i)
                self._summ_bytes -= old_length
            else:
                break
        kbps = self._summ_bytes * 8 / 1024 / self._sec_interval
        msg_data["kbps"] = kbps

        self._actual_list.append(msg_data)
        if len(self._actual_list) > self.actual_length:
            self._actual_list.pop(0)

    def add_packet(self, packet_data):
        """Добавить данные пакета."""

        msg_data = packet_data["msg_data"]

        frame_num = msg_data["frame_num"]
        seq_num = msg_data["seq_num"]
        payload = msg_data.pop("payload")
        is_exist = frame_num in self._pending_dict

        if is_exist:
            self._pending_dict[frame_num]["payload"][seq_num] = payload

            self._pending_dict[frame_num]["total_seq"] -= 1
            if self._pending_dict[frame_num]["total_seq"] == 0:
                all_payload = b''
                i = 0
                while True:
                    if i not in self._pending_dict[frame_num]["payload"]:
                        break

                    all_payload += self._pending_dict[frame_num]["payload"][i]

                    i += 1
                msg_data = self._pending_dict.pop(frame_num)
                self._pending_deque.remove(frame_num)
                self._len_counter -= 1

                msg_data["payload"] = all_payload
                del msg_data["total_seq"]
                del msg_data["seq_num"]
                self.add_actual_msg(msg_data)

        else:
            msg_data["total_seq"] -= 1

            if msg_data["total_seq"] == 0:
                del msg_data["total_seq"]
                del msg_data["seq_num"]
                msg_data["payload"] = payload

                self.add_actual_msg(msg_data)
            else:
                msg_data["payload"] = {
                    seq_num: payload
                }

                if self._len_counter == self.pending_length:
                    last_frame_num = self._pending_deque.pop()
                    del self._pending_dict[last_frame_num]
                else:
                    self._len_counter += 1
                self._pending_deque.appendleft(frame_num)
                self._pending_dict[frame_num] = msg_data


class StatMaster:
    def __init__(self, source_dataset_dir="record_frames", statfile="stand1_decoder_stat.csv", image_format=".png", is_utc=True,
                 record=False):
        self.source_dir = source_dataset_dir
        self.image_format = image_format
        self._statfiler = open(statfile, mode='w', encoding='utf-8', newline='')
        self.statfile = csv.writer(self._statfiler, delimiter=',')
        self.record = record

        if self.record:
            self.statfile.writerow(["index", "frame_num", "timestamp", "ssim", "mse", "psnr", "msize"])
        else:
            self.statfile.writerow(["index", "frame_num", "timestamp"])

        self._is_utc = is_utc
        self._i = 0

    def read_frame(self, frame_num):
        """Прочитать исходную картинку с cv2."""

        imfile = "{}{}".format(frame_num, self.image_format)
        impath = os.path.join(self.source_dir, imfile)
        if os.path.isfile(impath):
            frame = cv2.imread(impath)
        else:
            frame = None

        return frame

    def mse_metric(self, frame_num, image):
        """Расчёт метрики MSE."""

        real_image = self.read_frame(frame_num)
        if not real_image:
            return '-'
        mse = np.mean((image - real_image) ** 2)

        return mse

    def ssim_metric(self, frame_num, image):
        """Расчёт метрики SSIM."""

        real_image = self.read_frame(frame_num)
        if not real_image:
            return '-'
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)

        score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

        return score

    def psnr_metric(self, frame_num, image):
        """Расчёт метрики PSNR."""

        mse = self.mse_metric(frame_num, image)
        if mse == '-':
            return mse
        if mse == 0:
            return 100

        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

        return psnr

    def get_timestamp(self):
        """Временная метка."""

        if self._is_utc:
            now = datetime.utcnow()
        else:
            now = datetime.now()
        timestamp = datetime.isoformat(now)

        return timestamp

    def write_stat(self, frame_num, frame, cfg_num, kbps, msize):
        """Записать статистику."""

        if self.record:
            ssim = self.ssim_metric(frame_num, frame)
            mse = self.mse_metric(frame_num, frame)
            psnr = self.psnr_metric(frame_num, frame)
        timestamp = self.get_timestamp()

        if self.record:
            self.statfile.writerow([self._i, frame_num, timestamp, ssim, mse, psnr, msize])
        else:
            self.statfile.writerow([self._i, frame_num, timestamp])
        self._statfiler.flush()
        self._i += 1

    def brand_frame(self, frame, frame_num, kbps):
        """Клеймить кадр."""

        this_time = datetime.utcnow()
        text = this_time.strftime("%H:%M:%S:%f")

        height, width, _ = frame.shape
        used_font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3.0
        pos = [width // 2 - 27 * len(text), height // 2]
        color = [(0, 0, 0), (255, 255, 255)]
        thickness = [8, 3]

        for clr, thcn in zip(color, thickness):
            frame = cv2.putText(frame, text, pos, used_font,
                                font_scale, clr, thcn, cv2.LINE_AA)

        text = "Frame: {}".format(frame_num)
        pos = [width // 2 - 27 * len(text), height // 2 + 75]
        for clr, thcn in zip(color, thickness):
            frame = cv2.putText(frame, text, pos, used_font,
                                font_scale, clr, thcn, cv2.LINE_AA)

        return frame

    def __del__(self):
        self._statfiler.flush()
        self._statfiler.close()


class FrameManagerProcess(multiprocessing.Process):
    time_wait = 0.050

    def __init__(self, actual_list, source_dir, this_maxsize,
                 dest_dir="dest_frames", verbose=True, record=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cfg_guard = None  # ConfigurationGuardian()
        self._stat_master = None
        self._actual_list = actual_list
        self._verbose = verbose
        self._source_dir = source_dir
        self._record = record
        self._dest_dir = dest_dir
        self._this_maxsize = this_maxsize
        if self._record:
            if os.path.isdir(self._dest_dir):
                shutil.rmtree(self._dest_dir, ignore_errors=True)
            os.makedirs(self._dest_dir, exist_ok=True)

    def run(self):
        """Запуск процесса."""

        self._stat_master = StatMaster(self._source_dir, record=self._record)
        self._cfg_guard = ConfigurationGuardian(self._this_maxsize, enable_encoder=False, enable_decoder=True)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # cv2.destroyAllWindows()

        if self._verbose:
            no_frame_flag = True
            print("\n=== Запуск имитатора FPV-CTVP-сервера ===\n")

        cv2.namedWindow('DECODER', cv2.WINDOW_NORMAL)
        while True:
            msg_data = self._get_actual_message()
            if msg_data is None:
                if self._verbose:
                    if no_frame_flag:
                        print("\n=== Кадр отсутствует ===\n")
                        no_frame_flag = False
                time.sleep(self.time_wait)
                continue

            if self._verbose:
                no_frame_flag = True
                print("\n=== Кадр {} ===\n".format(msg_data["frame_num"]))

            kbps = msg_data["kbps"]
            cfg_num = msg_data["cfg_num"]
            frame_num = msg_data["frame_num"]
            neuro_codec = self._cfg_guard.get_configuration(cfg_num)
            if not neuro_codec:
                print("!!! Wrong configuration: {} !!!".format(cfg_num))
                continue
            payload = msg_data["payload"]
            dest_height = msg_data["height"]
            dest_width = msg_data["width"]

            msize = len(payload)

            frame = neuro_codec.decode_frame(payload,
                                             dest_height=dest_height,
                                             dest_width=dest_width)

            # new_frame = self._stat_master.brand_frame(frame, frame_num, kbps)
            self._stat_master.write_stat(frame_num, frame, cfg_num, kbps, msize)
            if self._record:
                write_path = os.path.join(self._dest_dir, str(frame_num) + ".png")
                cv2.imwrite(write_path, frame)
            cv2.imshow("DECODER", frame)
            cv2.waitKey(1)

            # cv2.imshow("=== STAND 1 DECODER ===", new_frame)
            # cv2.waitKey(1)
        cv2.destroyAllWindows()

    def _get_actual_message(self):
        """Получение актуального сообщения."""

        try:
            actual_msg = self._actual_list.pop()
            if actual_msg:
                return actual_msg
            else:
                return None
        except IndexError:
            return None


class FPV_CTVP_Server:
    len_limit = 30 + 3 * 720 * 1280
    max_tries = 1_000 * 5
    try_wait = 0.001
    data_wait = 0.001
    connection_reset_wait = 0.5

    def __init__(self, source_dir, this_maxsize, traceback_mode=False, payload_length=1300, port=6571, actual_length=1,
                 pending_length=10, record=False):
        self._payload_length = payload_length
        self._traceback_mode = traceback_mode
        self._source_dir = source_dir

        self.parser = PacketManager(version=0)

        self._internal_actual_list = []
        self._state_manager = multiprocessing.Manager()
        self._actual_list = self._state_manager.list(self._internal_actual_list)
        self._record = record

        self._packet_accounter = PacketAccounter(self._actual_list,
                                                 actual_length=actual_length,
                                                 pending_length=pending_length)
        self._frame_processor = FrameManagerProcess(self._actual_list, self._source_dir, this_maxsize,
                                                    record=self._record, daemon=True)

        signal.signal(signal.SIGINT, self.disable_server)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0)
        self._host = "0.0.0.0"
        self._port = port

    def run_server(self):
        """Запуск сервера."""

        self._frame_processor.start()
        self._socket.bind((self._host, self._port))
        packet_bytes = b''

        while True:
            try:
                try:
                    packet_bytes, source = self._socket.recvfrom(self._payload_length + 43)
                except socket.error:
                    time.sleep(self.data_wait)
                    continue
                packet_data = self.parser.parse_packet(packet_bytes)

                self._packet_accounter.add_packet(packet_data)
            except Exception as ex:
                if self._traceback_mode:
                    print(traceback.format_exc())
                else:
                    print(ex)
                self.restore_state()
                continue

    def restore_state(self):
        """Восстановление сервера при ошибке пакета.
        Чтение всех байтов (возможно, частичных) для перехода к новым сообщениям."""

        while True:
            try:
                result, _ = self._socket.recvfrom(65536)
                if result is None:
                    break
            except socket.error:
                break

    def disable_server(self, *_):
        """Безопасное выключение сервера."""

        # self._frame_processor.terminate()
        self._socket.close()

        raise KeyboardInterrupt()


def main():
    this_maxsize = 37_580_963_840

    print("\n=== Инициализация имитатора FPV-CTVP-сервера для стенда 1 ===\n")
    server = FPV_CTVP_Server("dataset_preparation/source_dataset", this_maxsize, traceback_mode=True, record=record,
                             payload_length=partition)
    try:
        server.run_server()
    except KeyboardInterrupt:
        print("\n=== Завершение имитатора FPV-CTVP-сервера ===")
        sys.exit()


if __name__ == "__main__":
    main()
