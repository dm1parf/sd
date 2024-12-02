import csv
import argparse
from datetime import datetime
import numpy as np

systematic_error = 0  # ms

encoder_file = "stand1_encoder_stat.csv"
decoder_file = "stand1_decoder_stat.csv"
parser = argparse.ArgumentParser(prog="Измеритель прикладной задержки", description="Измеряет задержку прикладного уровня")
parser.add_argument('-i', '--input', dest="input", type=str, default=encoder_file)
parser.add_argument('-o', '--output', dest="output", type=str, default=decoder_file)
arguments = parser.parse_args()
encoder_file = arguments.input
decoder_file = arguments.output

encoder_frames = dict()
decoder_frames = dict()

with (open(encoder_file, newline='', mode='r') as enc_file,
      open(decoder_file, newline='', mode='r') as dec_file
):
    enc_csv = csv.reader(enc_file)
    dec_csv = csv.reader(dec_file)
    enc_csv.__next__()
    dec_csv.__next__()

    for enc_record in enc_csv:
        frame_num, timestamp, *_ = enc_record
        td = datetime.fromisoformat(timestamp)
        encoder_frames[frame_num] = td

    for dec_record in dec_csv:
        ind, frame_num, timestamp, *_ = dec_record
        td = datetime.fromisoformat(timestamp)
        decoder_frames[frame_num] = td

enc_set = set(encoder_frames.keys())
dec_set = set(decoder_frames.keys())
diff_set = enc_set - dec_set

source_len = len(enc_set)
diff_len = len(diff_set)
went_wrong = diff_len
losses = round(diff_len / source_len * 100, 2)

full_set = enc_set & dec_set

latencies = []
for frame_num in full_set:
    latency = decoder_frames[frame_num] - encoder_frames[frame_num]
    latency = latency.total_seconds() * 1_000 + systematic_error
    latencies.append(latency)

latencies = np.array(latencies, dtype=np.float32)
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
