import csv
import argparse
import numpy as np

systematic_error = 0  # ms

encoder_file = "statlat_100.csv"
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
        id, payload_size, latency, is_good = enc_record

        all_count += 1

        is_good = bool(int(is_good))
        latency = float(latency)

        if not is_good:
            loss_count += 1
        else:
            all_latency.append(latency)


losses = loss_count / all_count * 100

if len(all_latency) == 0:
    print("=== !!! ===")
    print("Losses: 100 %")
    print("Cannot Count Mean Latency!")
    print("=== !!! ===")
else:
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
    print("Losses: {} %".format(losses))
    print("Mean Latency: {} ms".format(mean_latency))
    print("=== !!! ===")
