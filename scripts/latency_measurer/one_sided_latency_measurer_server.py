import socket
import signal
import csv
import struct
from datetime import datetime

stat_file = "one_sided_lms_serv_stat.csv"
socket_host = "0.0.0.0"
socket_port = 6565
byte_start = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'
len_start = len(byte_start)
byte_end = b'\x10\x09\x08\x07\x06\x05\x04\x03\x02\x01'
len_end = len(byte_end)
partition = 1_000
segment_wait = 0.001

new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
new_socket.bind((socket_host, socket_port))


def urgent_close(*_):
    global new_socket

    new_socket.close()
    exit()


def get_date():
    dater = datetime.utcnow()
    dater_str = dater.isoformat()
    return dater_str


signal.signal(signal.SIGINT, urgent_close)

with open(stat_file, mode='w', newline='') as stat_:
    csv_stat = csv.writer(stat_, delimiter=',')
    csv_stat.writerow(["id", "utc_date", "payload_size"])

    i = 0

    while True:
        try:
            frames_dict = dict()
            max_key = 0
            while True:
                new_byte, address = new_socket.recvfrom(65535)
                start_len = struct.unpack('I', new_byte[:4])[0]

                if start_len > max_key:
                    max_key = start_len

                bpartition = new_byte[4:]

                if bpartition[:len_start] == byte_start:
                    bpartition = bpartition[len_start:]

                if bpartition[-len_end:] == byte_end:
                    frames_dict[start_len] = bpartition[:-len_end]
                    break
                else:
                    frames_dict[start_len] = bpartition
            payload = b''
            for j in range(max_key + 1):
                payload += frames_dict[j]
            this_date = get_date()

            partition_num = 0
            window_start = 0
            window_end = partition

            payload_size = len(payload)

            csv_stat.writerow([i, this_date, payload_size])
            i += 1


        except:
            continue


