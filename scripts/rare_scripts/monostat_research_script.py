# import statsmodels.api as sm
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import hurst
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr
from scipy.integrate import simpson
# import scipy.stats as st
# import statsmodels.stats.api as sms
# from scipy.special import erf


warnings.filterwarnings("ignore")


go_params = True
go_kde = False
go_pearson = False
go_remake = True

# use_configs = range(1, 7)  # 1, 2, 3, 4, 5, 6.
use_configs = range(1, 6)  # 1, 2, 3, 4, 5.
conf_dict = {1: 1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 14}
word_dict = {1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е"}
word2_dict = {1: "b", 2: "v", 3: "g", 4: "d", 5: "e", 6: "zh"}
base_datapath = r"D:\UserData\Работа\Проекты_статей\Моностатья\stand_experiments\quality_neuro_HD_{}.csv"
msize_datapath = r"D:\UserData\Работа\Проекты_статей\Моностатья\stand_experiments\stat_decoder_HD_30112024_{}.csv"

dataset = {}
for new_conf_num in use_configs:
    conf_num = conf_dict[new_conf_num]

    conf_datapath = base_datapath.format(conf_num)
    conf2_datapath = msize_datapath.format(conf_num)
    conf_dataset = pd.read_csv(conf_datapath)
    msize_dataset = pd.read_csv(conf2_datapath)
    conf_dataset["msize"] = msize_dataset["msize"]
    dataset[new_conf_num] = conf_dataset


def formal_str(prm_mean, prm_var):
    return "N({}; {})".format(prm_mean, prm_var)


if go_params:
    all_ssim = []
    for conf_num in use_configs:
        conf_dataset = dataset[conf_num]
        ssim_mean = round(conf_dataset["ssim"].mean(), 2)
        psnr_mean = round(conf_dataset["psnr"].mean(), 2)
        msize_mean = round(conf_dataset["msize"].mean(), 2)
        ssim_var = round(conf_dataset["ssim"].var(), 4)
        psnr_var = round(conf_dataset["psnr"].var(), 4)
        msize_var = round(conf_dataset["msize"].var(), 4)

        pre_str = "{}\t{}\t{}\t{}".format(conf_num, formal_str(ssim_mean, ssim_var),
                                      formal_str(psnr_mean, psnr_var), formal_str(msize_mean, msize_var))
        final_str = pre_str.replace(".", ",")

        print(final_str)

        all_ssim.append(ssim_mean)

if go_kde:
    dest_prm = "msize"
    labeler = "MSize, байт"
    saver = r"D:\UserData\Работа\Проекты_статей\Моностатья\Рисунки\5{}.jpg"

    for conf_num in use_configs:
        conf_word = word_dict[conf_num]
        save_filepath = saver.format(conf_word)

        conf_dataset = dataset[conf_num]
        dest_series = conf_dataset[dest_prm].to_numpy()

        band = 75 + 50 * (conf_num - 1) * (conf_num < 5) + 50 * (conf_num == 3) + 35 * (conf_num > 4)
        # silverman 2.5
        # kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
        kde = KernelDensity(kernel="gaussian", bandwidth=band)
        very_dataset = dest_series
        very_dataset = very_dataset.reshape(-1, 1)

        kde = kde.fit(very_dataset)

        step = 0.01
        check = np.arange(very_dataset.min(), very_dataset.max() + step, step)
        check_ = check.reshape(-1, 1)
        y_data = np.exp(kde.score_samples(check_))  # * 100

        z = simpson(y_data, x=check)
        print("{}-Check:".format(conf_num), z)

        plt.xlabel(labeler)
        plt.ylabel("Плотность вероятности")
        # plt.plot(check, y_data, color='black')
        plt.plot(check, y_data)
        # plt.show()
        plt.savefig(save_filepath, dpi=300)
        plt.close()

if go_pearson:
    y = list(use_configs)
    res = pearsonr(all_ssim, y)
    print("r(ssim,№) =", res.statistic)

if go_remake:
    import os
    # source_img = r"D:\UserData\Работа\Проекты_статей\Моностатья\Рисунки\6а.jpg"
    os.chdir(r"D:\UserData\Работа\Сжатие_изображений\sd")
    source_img = "1.jpg"
    frame = cv2.imread(source_img)
    basic_size = (1280, 720)
    from scripts.stand1.stand1_decoder import ConfigurationGuardian
    cfg_guard = ConfigurationGuardian()

    # saver = r"D:\UserData\Работа\Проекты_статей\Моностатья\Рисунки\6{}.jpg"
    saver = r"6{}.jpg"
    for conf_num in use_configs:
        old_conf = conf_dict[conf_num]
        conf_word = word2_dict[conf_num]
        conf_saver = saver.format(conf_word)

        neuro_codec = cfg_guard.get_configuration(old_conf)

        latent = neuro_codec.encode_frame(frame)
        new_frame = neuro_codec.decode_frame(latent, dest_height=basic_size[1], dest_width=basic_size[0])
        print(conf_saver, new_frame.shape)
        cv2.imwrite(conf_saver, new_frame)



