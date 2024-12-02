import csv
import argparse
from datetime import datetime
import numpy as np

systematic_error = 0  # ms

decoder_file = "stand1_decoder_stat.csv"
parser = argparse.ArgumentParser(prog="Измеритель MSize", description="Измеряет размер сжатого представления")
parser.add_argument('-i', '--input', dest="input", type=str, default=decoder_file)
arguments = parser.parse_args()
decoder_file = arguments.input

decoder_frames = dict()

msizes = []
with open(decoder_file, newline='', mode='r') as dec_file:
    dec_csv = csv.reader(dec_file)
    dec_csv.__next__()

    for dec_record in dec_csv:
        index, frame_num, timestamp, ssim, mse, psnr, msize, *_ = dec_record
        msizes.append(msize)

msizes = np.array(msizes, dtype=np.float32)
K1 = np.quantile(msizes, 0.25)
K3 = np.quantile(msizes, 0.75)
DK = K3 - K1
min_msize = K1 - 1.5 * DK
max_msize = K3 + 1.5 * DK
msizes = msizes[msizes > min_msize]
msizes = msizes[msizes < max_msize]

mean_msize = msizes.mean()

print("==========")
print("MSizeср: {} байт".format(mean_msize))
print("==========")
