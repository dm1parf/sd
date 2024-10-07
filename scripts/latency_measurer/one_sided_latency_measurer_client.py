import csv
import socket
import argparse
from datetime import datetime
import time
import signal
import struct
import math

stat_fn = "one_sided_lms_client_stat.csv"
address = "127.0.0.1"
# 91.238.230.84
# 127.0.0.1
port = 6560
number_of_packets = 50
payload_size = 10000
timeout = 10
fps = 15
between_wait = 1 / fps
partition = 1_000
every_progress = 100
segment_wait = 0.001
# В байтах (!!!)

parser = argparse.ArgumentParser(prog="Измеритель прикладной задержки", description="Измеряет задержку прикладного уровня")
parser.add_argument('-n', '--number', dest="number", type=int, default=number_of_packets)
parser.add_argument('-s', '--payload', dest="payload", type=int, default=payload_size)
parser.add_argument('-f', '--statfile', dest="statfile", type=str, default=stat_fn)
parser.add_argument('-i', '--ip', dest="ip", type=str, default=address)
parser.add_argument('-p', '--port', dest="port", type=int, default=port)
parser.add_argument('-c', '--cut', dest="cut", type=int, default=partition)
arguments = parser.parse_args()

number_of_packets = arguments.number
payload_size = arguments.payload
address = arguments.ip
port = arguments.port
partition = arguments.cut
total_seq = math.ceil(payload_size / partition)
print(total_seq)

stat_fn = arguments.statfile
stat_file = open(stat_fn, 'w', newline='')
csv_stat = csv.writer(stat_file)
csv_stat.writerow(["id", "payload_size", "utc_date"])

this_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
this_socket.settimeout(timeout)


def get_date():
    dater = datetime.utcnow()
    dater_str = dater.isoformat()
    return dater_str


def urgent_close(*_):
    global this_socket
    global stat_file

    stat_file.close()
    this_socket.close()


signal.signal(signal.SIGINT, urgent_close)

print("=== Начинаем испытания: {} UDP-датаграмм с полезной нагрузкой {} байт".format(number_of_packets, payload_size))

for i in range(number_of_packets):
    time.sleep(between_wait)
    if (i % every_progress) == 0:
        print("{}/{}".format(i+1, number_of_packets))

    start = datetime.now()
    # try
    if True:
        original_payload = bytes(payload_size)
        # payload = byte_start + original_payload + byte_end
        payload = original_payload
        partition_num = 0
        window_start = 0
        window_end = partition
        true_payload_size = len(payload)
        while window_end < true_payload_size:
            part_bytes = struct.pack('>III', i, partition_num, total_seq)
            part_payload = part_bytes + payload[window_start:window_end]
            this_socket.sendto(part_payload, (address, port))
            window_start = window_end
            window_end += partition
            partition_num += 1
            time.sleep(segment_wait)
        else:
            if window_start < true_payload_size:
                part_bytes = struct.pack('>III', i, partition_num, total_seq)
                part_payload = part_bytes + payload[window_start:]
                this_socket.sendto(part_payload, (address, port))
        this_date = get_date()

        print(i, payload_size, this_date)
        csv_stat.writerow([i, payload_size, this_date])

urgent_close()
print("=== Успешно завершено! ===")
