import csv
import argparse
from datetime import datetime
import numpy as np

systematic_error = 0  # ms

# encoder_file = "statlat_100.csv"
encoder_file = r"C:\Users\Александр\Downloads\ocr_test_h265_380.csv"
parser = argparse.ArgumentParser(prog="Измеритель прикладной задержки", description="Измеряет задержку прикладного уровня")
parser.add_argument('-i', '--input', dest="input", type=str, default=encoder_file)
arguments = parser.parse_args()
encoder_file = arguments.input

encoder_frames = dict()

all_count = 0
loss_count = 0
all_latency = []

with open(encoder_file, newline='', mode='r') as enc_file:
    enc_csv = csv.reader(enc_file)
    enc_csv.__next__()

    for enc_record in enc_csv:
        frame_num, latency, _, _ = enc_record
        latency = float(latency)
        all_latency.append(latency)

latencies = np.array(all_latency, dtype=np.float32)
K1 = np.quantile(latencies, 0.25)
K3 = np.quantile(latencies, 0.75)
DK = K3 - K1
min_latency = K1 - 1.5 * DK
max_latency = K3 + 1.5 * DK
latencies = latencies[latencies > min_latency]
latencies = latencies[latencies < max_latency]

mean_latency = latencies.mean()

print("=== !!! ===")
print("Mean Latency: {} ms".format(mean_latency))
print("=== !!! ===")
