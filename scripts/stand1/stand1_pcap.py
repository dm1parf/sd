import pyshark
import argparse
import csv
from datetime import datetime

csv_file = "stat_pcap.csv"
source_pcap = "usb_camera_08.10.24.pcap"

arguments = argparse.ArgumentParser(prog="Анализатор PCAP",
                                    description="Сделано для испытаний пропускной способности канала.")
arguments.add_argument("-p", "--pcap", dest="pcap", type=str, default=source_pcap, help="Файл PCAP")
arguments.add_argument("-o", "--output", dest="output", type=str, default=csv_file, help="Файл CSV")
args = arguments.parse_args()
source_pcap = args.pcap
csv_file = args.output

cap = pyshark.FileCapture(source_pcap)

time_first = None
time_last = None
total_len = 0

frame_num = 0
with open(csv_file, mode='w', newline='') as cf:
    csv_writer = csv.writer(cf)
    csv_writer.writerow(["frame_num", "length", "datetime"])

    for packet in cap:
        frame_data = packet.frame_info
        time_epoch = float(frame_data.time_epoch)

        time_utc = datetime.fromtimestamp(time_epoch)
        if not time_first:
            time_first = time_utc
        time_last = time_utc
        time_utc_str = time_utc.isoformat()
        frame_size = int(frame_data.len)

        print(frame_num, frame_size, time_utc_str)
        csv_writer.writerow([frame_num, frame_size, time_utc_str])

        total_len += frame_size
        frame_num += 1

delter = (time_last - time_first).total_seconds()
kbps = total_len * 8 / 1024 / delter
kbps = round(kbps, 2)

print("=========")
print("Bitrate = {} kbps".format(kbps))
print("=========")




