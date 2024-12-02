import socket
import signal
import csv
import struct
from collections import deque
from datetime import datetime
import traceback
import time
import sys

stat_file = "one_sided_lms_serv_stat.csv"
socket_host = "0.0.0.0"
socket_port = 6560
# byte_start = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'
# len_start = len(byte_start)
# byte_end = b'\x10\x09\x08\x07\x06\x05\x04\x03\x02\x01'
# len_end = len(byte_end)
partition = 1_000
segment_wait = 0.001

new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
new_socket.bind((socket_host, socket_port))


class PacketAccounter:
    def __init__(self, actual_list, pending_length=10, actual_length=5):
        self._pending_dict = dict()
        self._pending_deque = deque(maxlen=pending_length)  # FIFO
        # self._actual_deque = deque(maxlen=actual_length)  # LIFO
        self._internal_actual_list = []
        self._actual_list = actual_list
        self._stat = open(stat_file, mode='w', newline='')
        self._csv_stat = csv.writer(self._stat, delimiter=',')
        self._csv_stat.writerow(["id", "utc_date", "payload_size"])

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
        timestamp = now_datetime.isoformat()
        frame_num = msg_data["frame_num"]
        payload_size = len(msg_data["payload"])

        print(frame_num, timestamp, payload_size)
        self._csv_stat.writerow([frame_num, timestamp, payload_size])

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

    def stat_close(self):
        self._stat.flush()
        self._stat.close()


class OneSidedPacketParser:
    def __init__(self):
        pass

    def parse_packet(self, packet):
        """Разобрать пакет FPV-CTVP."""

        header = packet[:12]
        payload = packet[12:]

        frame_num, seq_num, total_seq = struct.unpack(">III", header)

        packet_data = dict()
        msg_data = dict()

        msg_data["frame_num"] = frame_num
        msg_data["seq_num"] = seq_num
        msg_data["total_seq"] = total_seq
        msg_data["payload"] = payload

        packet_data["msg_data"] = msg_data

        return packet_data


class OneSidedServer:
    len_limit = 30 + 3 * 720 * 1280
    max_tries = 1_000 * 5
    try_wait = 0.001
    data_wait = 0.001
    connection_reset_wait = 0.5

    def __init__(self, source_dir, traceback_mode=False, payload_length=1000, port=6560, actual_length=1,
                 pending_length=10):
        self._payload_length = payload_length
        self._traceback_mode = traceback_mode
        self._source_dir = source_dir

        self.parser = OneSidedPacketParser()

        self._internal_actual_list = []
        self._actual_list = []
        self._packet_accounter = PacketAccounter(self._actual_list,
                                                 actual_length=actual_length,
                                                 pending_length=pending_length)
        signal.signal(signal.SIGINT, self.disable_server)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0)
        self._host = "0.0.0.0"
        self._port = port

    def run_server(self):
        """Запуск сервера."""

        self._socket.bind((self._host, self._port))
        packet_bytes = b''

        while True:
            try:
                try:
                    packet_bytes, source = self._socket.recvfrom(1020)
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
        self._packet_accounter.stat_close()

        raise KeyboardInterrupt()


def main():
    print("\n=== Инициализация одностороннего скрипта измерения задержек ===\n")
    server = OneSidedServer("dataset_preparation/source_dataset", traceback_mode=True)
    try:
        server.run_server()
    except KeyboardInterrupt:
        print("\n=== Завершение одностороннего скрипта измерения задержек ===")
        sys.exit()


if __name__ == "__main__":
    main()
