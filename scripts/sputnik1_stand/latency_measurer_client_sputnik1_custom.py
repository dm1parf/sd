import csv
import socket
import argparse
import datetime
import time
import signal
import struct

stat_fn = "statlat_{}.csv"
address = "91.238.230.83"
# 91.238.230.84
# 127.0.0.1
port = 6565
number_of_packets = 1_000
payload_size_all = [100, 300, 500, 700, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 15000]
timeout = 5
between_wait = 0.05
partition = 1_000
every_progress = 100
segment_wait = 0.001
# В байтах (!!!)
byte_start = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10'
len_start = len(byte_start)
byte_end = b'\x10\x09\x08\x07\x06\x05\x04\x03\x02\x01'
len_end = len(byte_end)
minpayload = 0

parser = argparse.ArgumentParser(prog="Измеритель прикладной задержки", description="Измеряет задержку прикладного уровня")
parser.add_argument('-i', '--ip', dest="ip", type=str, default=address)
parser.add_argument('-p', '--port', dest="port", type=int, default=port)
parser.add_argument('-c', '--cut', dest="cut", type=int, default=partition)
parser.add_argument('--payloads', dest="payloads", nargs="+", type=int, default=[])
parser.add_argument('--numpackets', dest="numpackets", type=int, default=number_of_packets)
arguments = parser.parse_args()

address = arguments.ip
port = arguments.port
partition = arguments.cut
number_of_packets = arguments.numpackets
payload_size_all = arguments.payloads
print("Полезные нагрузки:", *payload_size_all)

for payload_size in payload_size_all:
    stat_filename = stat_fn.format(payload_size)

    with open(stat_filename, 'w', newline='') as stat_file:
        csv_stat = csv.writer(stat_file)
        csv_stat.writerow(["id", "payload_size", "latency", "is_good"])

        this_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        this_socket.settimeout(timeout)

        out_of_order_starts = dict()
        out_of_order_ends = dict()


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

            start = datetime.datetime.now()
            try:
                frame_num = struct.pack('I', i)

                original_payload = bytes(payload_size)
                payload = byte_start + frame_num + original_payload + byte_end
                partition_num = 0
                window_start = 0
                window_end = partition
                true_payload_size = len(payload)
                while window_end < true_payload_size:
                    part_bytes = struct.pack('I', partition_num)
                    part_payload = part_bytes + payload[window_start:window_end]
                    this_socket.sendto(part_payload, (address, port))
                    window_start = window_end
                    window_end += partition
                    partition_num += 1
                    time.sleep(segment_wait)
                else:
                    if window_start < true_payload_size:
                        part_bytes = struct.pack('I', partition_num)
                        part_payload = part_bytes + payload[window_start:]
                        this_socket.sendto(part_payload, (address, port))

                frames_dict = dict()
                max_key = 0
                while True:
                    new_byte, _ = this_socket.recvfrom(65535)
                    start_len = struct.unpack('I', new_byte[:4])[0]

                    if start_len > max_key:
                        max_key = start_len

                    bpartition = new_byte[4:]

                    if bpartition[:len_start] == byte_start:
                        frame_num = struct.unpack('I', bpartition[len_start:len_start + 4])[0]
                        bpartition = bpartition[len_start + 4:]

                    if bpartition[-len_end:] == byte_end:
                        frames_dict[start_len] = bpartition[:-len_end]
                        break
                    else:
                        frames_dict[start_len] = bpartition

                returned_payload = b''
                for j in range(max_key + 1):
                    returned_payload += frames_dict[j]

                assert returned_payload == original_payload
                end = datetime.datetime.now()

                if i == frame_num:
                    latency = (end - start).total_seconds() / 2 * 1000
                    latency = round(latency, 4)
                    csv_stat.writerow([i, payload_size, latency, 1])
                    stat_file.flush()
                else:
                    out_of_order_starts[i] = start
                    out_of_order_ends[frame_num] = end
            except (ConnectionResetError, TimeoutError, AssertionError, KeyError) as err:
                # print("Ошибка:", err)
                csv_stat.writerow([i, payload_size, -1, 0])
                stat_file.flush()
            except OSError as err:
                # print("Ошибка2:", err)
                csv_stat.writerow([i, payload_size, -1, 0])
                stat_file.flush()

                this_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                this_socket.settimeout(timeout)

        # Исправление редкой проблемы с проблемой порядка кадров и зацепки за ложные кадрыIridium SBD

        keys_start = set(out_of_order_starts.keys())
        keys_end = set(out_of_order_ends.keys())
        common_keys = keys_start & keys_end
        uncommon_keys = (keys_start - common_keys) | (keys_end - common_keys)

        for key in common_keys:
            latency = (out_of_order_ends[key] - out_of_order_starts[key]).total_seconds() / 2 * 1000
            latency = round(latency, 4)
            csv_stat.writerow([i, payload_size, latency, 1])
        stat_file.flush()

        for key in uncommon_keys:
            csv_stat.writerow([i, payload_size, -1, 0])
        stat_file.flush()

print("=== Успешно завершено! ===")
