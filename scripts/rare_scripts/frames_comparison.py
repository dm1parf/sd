import os
import csv
import cv2
from utils.statistics import StatisticsManager


dir_base = "11072024_07_29"
one_dir = "{}_coder".format(dir_base)
two_dir = "{}_decoder".format(dir_base)
stat_dir = "scripts/rare_scripts"
stat_fn = "{}.csv".format(dir_base)
stat_filename = os.path.join(stat_dir, stat_fn)

list1 = os.listdir(one_dir)
list2 = os.listdir(two_dir)
list_all = list(set(list1) & set(list2))
list_all.sort(key=lambda x: int(os.path.splitext(x)[0]))

with open(stat_filename, newline='', mode='w', encoding='utf-8',) as cf:
    csv_writer = csv.writer(cf)
    csv_writer.writerow(["i", "filename", "ssim", "mse", "psnr"])

    for i, filename in enumerate(list_all):
        image_1 = cv2.imread(os.path.join(one_dir, filename))
        image_2 = cv2.imread(os.path.join(two_dir, filename))
        image_2 = cv2.resize(image_2, (640, 480), interpolation=cv2.INTER_AREA)

        ssim = StatisticsManager.ssim_metric(image_1, image_2)
        mse = StatisticsManager.mse_metric(image_1, image_2)
        psnr = StatisticsManager.psnr_metric(image_1, image_2)

        csv_writer.writerow([i, filename, ssim, mse, psnr])

