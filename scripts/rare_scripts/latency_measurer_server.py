import socket
import signal
import time
import struct

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


signal.signal(signal.SIGINT, urgent_close)


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

        partition_num = 0
        window_start = 0
        window_end = partition

        # payload
        payload = byte_start + payload + byte_end
        payload_size = len(payload)

        while window_end < payload_size:
            part_bytes = struct.pack('I', partition_num)
            part_payload = part_bytes + payload[window_start:window_end]
            new_socket.sendto(part_payload, address)
            window_start = window_end
            window_end += partition
            partition_num += 1
            time.sleep(segment_wait)
        else:
            if window_start < payload_size:
                part_bytes = struct.pack('I', partition_num)
                part_payload = part_bytes + payload[window_start:]
                new_socket.sendto(part_payload, address)
    except:
        continue


