import signal
import socket
import cv2
import torch
import os
import struct
import time
import sys
import csv
import traceback
import multiprocessing
from collections import deque
from skimage.metrics import structural_similarity
import numpy as np
import math

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from datetime import datetime
from utils.workers import (WorkerASDummy, WorkerASMoveDistribution, WorkerQuantLinear, WorkerAutoencoderKL_F16,
                           WorkerAutoencoderKL_F4, WorkerCompressorJpegXL, WorkerCompressorJpegXR,
                           WorkerCompressorAvif, WorkerCompressorDummy, WorkerQuantPower,
                           WorkerSRDummy)


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


class NeuroCodec:
    """Нейросетевой кодек: кодер + декодер."""

    def __init__(self, as_=None, vae=None, quant=None, compressor=None, sr=None):
        if as_:
            self._as = as_
        else:
            self._as = WorkerASDummy()

        self._quant = quant

        if compressor:
            self._compressor = compressor
        else:
            self._compressor = WorkerCompressorDummy()

        self._vae = vae
        if self._quant:
            self.dest_type = torch.uint8
        else:
            if self._vae:
                self.dest_type = self._vae.nominal_type
            else:
                self.dest_type = torch.float16
        if vae:
            self.dest_shape = self._vae.z_shape
        else:
            self.dest_shape = (1, 3, 512, 512)

        if sr:
            self._sr = sr
        else:
            self._sr = WorkerSRDummy()

    def decode_frame(self, binary):
        """Декодировать сжатое бинарное представление кадра."""

        with torch.no_grad():
            quant_latent, _ = self._compressor.decompress_work(binary,
                                                               dest_shape=self.dest_shape,
                                                               dest_type=self.dest_type)
            if self._quant:
                latent, _ = self._quant.dequant_work(quant_latent, dest_type=self._vae.nominal_type)
            else:
                latent = quant_latent
            if self._vae:
                image, _ = self._vae.decode_work(latent)
            else:
                image = latent
            frame, _ = self._as.restore_work(image)
            restored_frame, _ = self._sr.sr_work(frame, dest_size=[1080, 720])

        return restored_frame

    def encode_frame(self, frame):
        """Кодировать кадр в сжатую бинарную последовательность."""

        with torch.no_grad():
            image, _ = self._as.prepare_work(frame)
            if self._vae:
                latent, _ = self._vae.encode_work(image)
            else:
                latent = image
            if self._quant:
                (quant_latent, _), _ = self._quant.quant_work(latent)
            else:
                quant_latent = latent
            binary, _ = self._compressor.compress_work(quant_latent)

        return binary


class ConfigurationGuardian:
    def __init__(self):
        # Определение конфигураций

        # as_ = WorkerASDummy()
        as_ = WorkerASMoveDistribution()

        kl_f4 = WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml",
                                       ckpt_path="dependence/ckpt/kl-f4.ckpt")
        kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                         ckpt_path="dependence/ckpt/kl-f16.ckpt")

        quant_lin_bitround1_klf4 = WorkerQuantLinear(pre_quant="bitround", nsd=1)
        quant_lin_bitround1_klf4.adjust_params(autoencoder_worker="AutoencoderKL_F4")
        quant_lin_scale1_klf4 = WorkerQuantLinear(pre_quant="scale", nsd=1)
        quant_lin_scale1_klf4.adjust_params(autoencoder_worker="AutoencoderKL_F4")
        quant_lin_scale1_klf16 = WorkerQuantLinear(pre_quant="scale", nsd=1)
        quant_lin_scale1_klf16.adjust_params(autoencoder_worker="AutoencoderKL_F16")
        quant_lin_klf16 = WorkerQuantLinear()
        quant_lin_klf16.adjust_params(autoencoder_worker="AutoencoderKL_F16")
        quant_pow_scale1_klf16 = WorkerQuantPower(pre_quant="scale", nsd=1)
        quant_pow_scale1_klf16.adjust_params(autoencoder_worker="AutoencoderKL_F16")

        compress_jpegxl65 = WorkerCompressorJpegXL(65)
        compress_jpegxr55 = WorkerCompressorJpegXR(55)
        compress_jpegxr60 = WorkerCompressorJpegXR(60)
        compress_jpegxr65 = WorkerCompressorJpegXR(65)
        compress_jpegxr70 = WorkerCompressorJpegXR(70)
        compress_jpegxr75 = WorkerCompressorJpegXR(75)
        compress_jpegxr80 = WorkerCompressorJpegXR(80)
        compress_jpegxr85 = WorkerCompressorJpegXR(85)
        compress_avif75 = WorkerCompressorAvif(75)
        compress_avif80 = WorkerCompressorAvif(80)

        neuro_cfg1 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_bitround1_klf4,
                                compressor=compress_jpegxl65)
        neuro_cfg2 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr55)
        neuro_cfg3 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr60)
        neuro_cfg4 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr65)
        neuro_cfg5 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr70)
        neuro_cfg6 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr75)
        neuro_cfg7 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr80)
        neuro_cfg8 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr85)
        neuro_cfg9 = NeuroCodec(as_=as_,
                                vae=kl_f16,
                                quant=quant_lin_scale1_klf16,
                                compressor=compress_avif75)
        neuro_cfg10 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_lin_klf16,
                                 compressor=compress_avif80)
        neuro_cfg11 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_lin_klf16,
                                 compressor=compress_jpegxr70)
        neuro_cfg12 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr70)
        neuro_cfg13 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr75)
        neuro_cfg14 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr80)

        self._configurations = {
            1: neuro_cfg1,
            2: neuro_cfg2,
            3: neuro_cfg3,
            4: neuro_cfg4,
            5: neuro_cfg5,
            6: neuro_cfg6,
            7: neuro_cfg7,
            8: neuro_cfg8,
            9: neuro_cfg9,
            10: neuro_cfg10,
            11: neuro_cfg11,
            12: neuro_cfg12,
            13: neuro_cfg13,
            14: neuro_cfg14,
        }

    def get_configuration(self, cfg_num):
        """Получить нейросетевой кодек конфигурации."""

        cfg_data = self._configurations.get(cfg_num, None)

        return cfg_data


class PacketParser:
    supported_versions = (0,)

    def __init__(self, version):
        assert version in self.supported_versions, NotImplementedError("Wrong version:", version)

        self.version = version

        self.message_handlers = {
            5: self.parse_vstr,
        }

    def parse_packet(self, packet):
        """Разобрать пакет FPV-CTVP."""

        (version, stream_identifier, segment_type,
         message_type, message_length) = struct.unpack('>BHBBQ', packet[:13])
        message = packet[13:]

        assert len(message) == message_length, ValueError("Incorrect message length")

        message_data = self.parse_message(message_type, message)

        packet_data = {
            "msg_data": message_data,
            "version": version,
            "stream_id": stream_identifier,
            "segment_type": segment_type,
            "message_type": message_type
        }

        return packet_data

    def parse_message(self, message_type, message):
        """Разобрать сообщение FPV-CTVP."""

        message_handler = self.message_handlers.get(message_type, None)

        if message_handler is None:
            raise NotImplementedError("Wrong message type:", message_type)
        else:
            message_content = message_handler(message)
            return message_content

    def parse_vstr(self, message):
        """Разобрать сообщение VSTR FPV-CTVP."""

        (frame_num, segment_num, total_segments,
         height, width, cfg_num, encryption_num,
         payload_length) = struct.unpack(">QHHIIBBQ", message[:30])
        payload = message[30:]

        message_data = {
            "frame_num": frame_num,
            "seq_num": segment_num,
            "total_seq": total_segments,
            "payload": payload,
            "height": height,
            "width": width,
            "cfg_num": cfg_num,
            "encryption_num": encryption_num
        }

        return message_data


class StatMaster:
    def __init__(self, source_dataset_dir, statfile="stand1_decoder_stat.csv", image_format=".jpg", is_utc=True):
        self.source_dir = source_dataset_dir
        self.image_format = image_format
        self._statfiler = open(statfile, mode='w', encoding='utf-8', newline='')
        self.statfile = csv.writer(self._statfiler, delimiter=',')
        self.statfile.writerow(["index", "frame_num", "timestamp"])

        self._is_utc = is_utc
        self._i = 0

    def read_frame(self, frame_num):
        """Прочитать исходную картинку с cv2."""

        imfile = "{}{}".format(frame_num, self.image_format)
        impath = os.path.join(self.source_dir, imfile)
        frame = cv2.imread(impath)

        return frame

    def mse_metric(self, frame_num, image):
        """Расчёт метрики MSE."""

        real_image = self.read_frame(frame_num)
        mse = np.mean((image - real_image) ** 2)

        return mse

    def ssim_metric(self, frame_num, image):
        """Расчёт метрики SSIM."""

        real_image = self.read_frame(frame_num)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)

        score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

        return score

    def psnr_metric(self, frame_num, image):
        """Расчёт метрики PSNR."""

        mse = self.mse_metric(frame_num, image)
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

    def write_stat(self, frame_num, frame, cfg_num, kbps):
        """Записать статистику."""

        ssim = self.ssim_metric(frame_num, frame)
        mse = self.mse_metric(frame_num, frame)
        psnr = self.psnr_metric(frame_num, frame)
        timestamp = self.get_timestamp()

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

    def __init__(self, actual_list, source_dir, verbose=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cfg_guard = None  # ConfigurationGuardian()
        self._stat_master = None
        self._actual_list = actual_list
        self._verbose = verbose
        self._source_dir = source_dir

    def run(self):
        """Запуск процесса."""

        self._stat_master = StatMaster(self._source_dir)
        self._cfg_guard = ConfigurationGuardian()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cv2.destroyAllWindows()

        if self._verbose:
            no_frame_flag = True
            print("\n=== Запуск имитатора FPV-CTVP-сервера ===\n")

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

            frame = neuro_codec.decode_frame(payload)

            # new_frame = self._stat_master.brand_frame(frame, frame_num, kbps)
            self._stat_master.write_stat(frame_num, frame, cfg_num, kbps)

            # cv2.imshow("=== STAND 1 DECODER ===", new_frame)
            # cv2.waitKey(1)

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

    def __init__(self, source_dir, traceback_mode=False, payload_length=1000, port=6571, actual_length=1,
                 pending_length=10):
        self._payload_length = payload_length
        self._traceback_mode = traceback_mode
        self._source_dir = source_dir

        self.parser = PacketParser(version=0)

        self._internal_actual_list = []
        self._state_manager = multiprocessing.Manager()
        self._actual_list = self._state_manager.list(self._internal_actual_list)

        self._packet_accounter = PacketAccounter(self._actual_list,
                                                 actual_length=actual_length,
                                                 pending_length=pending_length)
        self._frame_processor = FrameManagerProcess(self._actual_list, self._source_dir, daemon=True)

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
    print("\n=== Инициализация имитатора FPV-CTVP-сервера для стенда 1 ===\n")
    server = FPV_CTVP_Server("dataset_preparation/source_dataset", traceback_mode=True)
    try:
        server.run_server()
    except KeyboardInterrupt:
        print("\n=== Завершение имитатора FPV-CTVP-сервера ===")
        sys.exit()


if __name__ == "__main__":
    main()
