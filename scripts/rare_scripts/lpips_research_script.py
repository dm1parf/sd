import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import math
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import re

storage_file = r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\pd_test.zip"
method_parser = re.compile(r"encoder_([^_]+)_.*")
standard_dirs = [
    ["encoder_av1_hd_25", "decoder_av1_hd_hd"],
    ["encoder_vp9_hd_25", "decoder_vp9_hd_hd"],
    ["encoder_h264_hd_25", "decoder_h264_hd_hd"],
    ["encoder_h265_hd_25", "decoder_h265_hd_hd"],
    ["encoder_h266_hd_25", "decoder_h266_hd_hd"],
]
standard_psnr = []
standard_mse = []
standard_ssim = []
standard_lpips_alex = []
standard_lpips_vgg = []

neuro_dirs = [
    ["encoder_nc1_hd_15", "decoder_nc1_hd_15"],
    ["encoder_nc2_hd_15", "decoder_nc2_hd_15"],
]
neuro_psnr = []
neuro_mse = []
neuro_ssim = []
neuro_lpips_alex = []
neuro_lpips_vgg = []


def convert_to_tensor(n):
    n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    n = np.moveaxis(n, 2, 0)
    t = torch.from_numpy(n).cuda()
    t = t.to(dtype=torch.float32)
    t -= (255 / 2)
    t /= (255 / 2)
    t = torch.clamp(t, min=-1.0, max=1.0)
    # Для LPIPS нужно масштабирование от -1 до 1!!!

    return t


def get_lpips_value(frame1, frame2):
    t1 = convert_to_tensor(frame1)
    t2 = convert_to_tensor(frame2)

    value_vgg = lpips_vgg.forward(t1, t2).item()
    value_alex = lpips_alex.forward(t1, t2).item()

    # 0 -- полное сходство
    # Больше -- потери
    return value_vgg, value_alex


# test_img1 = r"decoder_h264_hd_hd/frame_0010.png"
# test_img2 = r"encoder_h264_hd_25/frame_0010.png"

# img1 = cv2.imread(test_img1)
# img2 = cv2.imread(test_img2)

# with torch.no_grad():
#     lpips_vgg_val, lpips_alex_val = get_lpips_value(img1, img2)
# print(lpips_vgg_val, lpips_alex_val)


def mse_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики MSE.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = np.mean((image1 - image2) ** 2)

    return mse


def ssim_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики SSIM.
    На вход подаются две картинки в формате cv2 (numpy)."""

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image2, image1, data_range=image2.max() - image2.min())

    return score


def psnr_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """Расчёт метрики PSNR.
    На вход подаются две картинки в формате cv2 (numpy)."""

    mse = mse_metric(image1, image2)
    if mse == 0:
        return 100

    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def quality_measurer_neuro(encoder_dir, decoder_dir):
    psnr_table = []
    ssim_table = []
    mse_table = []
    lpips_vgg_table = []
    lpips_alex_table = []
    association_table = []

    this_enc_count = -1
    while True:
        this_enc_count += 1
        new_frame = "{}.png".format(this_enc_count)
        enc_file = os.path.join(encoder_dir, new_frame)
        dec_file = os.path.join(decoder_dir, new_frame)

        if not os.path.isfile(enc_file):
            break
        if not os.path.isfile(dec_file):
            continue
        enc_img = cv2.imread(enc_file)
        dec_img = cv2.imread(dec_file)

        assoc = (this_enc_count, this_enc_count)
        psnr_met = round(psnr_metric(enc_img, dec_img), 6)
        ssim_met = round(ssim_metric(enc_img, dec_img), 6)
        mse_met = round(mse_metric(enc_img, dec_img), 6)
        lpips_vgg_met, lpips_alex_met = get_lpips_value(enc_img, dec_img)
        lpips_vgg_met = round(lpips_vgg_met, 6)
        lpips_alex_met = round(lpips_alex_met, 6)

        psnr_table.append(psnr_met)
        mse_table.append(mse_met)
        ssim_table.append(ssim_met)
        lpips_vgg_table.append(lpips_vgg_met)
        lpips_alex_table.append(lpips_alex_met)
        association_table.append(assoc)

    new_association_table = [str(i[0]) + " " + str(i[1]) for i in association_table]
    result = pd.DataFrame(data={"association_table": new_association_table,
                                "psnr_table": psnr_table,
                                "mse_table": mse_table,
                                "ssim_table": ssim_table,
                                "lpips_vgg_table": lpips_vgg_table,
                                "lpips_alex_table": lpips_alex_table})

    return result


def quality_measurer_standard(encoder_dir, decoder_dir, max_dec_forward=200, max_enc_forward=50, mse_threshold=80.0):
    framer = "frame_{:04d}.png"

    def get_dir_min_max(direr):
        if not os.path.isdir(direr):
            return None, None

        dir_starter = None
        dir_ender = None
        for dir_count in range(10_000):
            namer = framer.format(dir_count)
            namepath = os.path.join(direr, namer)
            if (dir_ender is not None) and (not os.path.isfile(namepath)):
                break
            if os.path.isfile(namepath):
                if (dir_starter is None) and os.path.isfile(namepath):
                    dir_starter = dir_count
                dir_ender = dir_count

        return dir_starter, dir_ender

    def read_img_num(direr, number):
        namer = framer.format(number)
        pather = os.path.join(direr, namer)
        image = cv2.imread(pather)

        return image

    min_encoder_framer, max_encoder_framer = get_dir_min_max(encoder_dir)
    min_decoder_framer, max_decoder_framer = get_dir_min_max(decoder_dir)
    enc_frame_num = max_encoder_framer - min_encoder_framer + 1
    dec_frame_num = max_decoder_framer - min_decoder_framer + 1

    association_table = []
    psnr_table = []
    ssim_table = []
    mse_table = []
    lpips_vgg_table = []
    lpips_alex_table = []

    decoder_frame_cache = dict()
    encoder_frame_cache = dict()
    cur_dec_frame = min_decoder_framer
    # for cur_enc_frame in range(min_encoder_framer, max_encoder_framer + 1):
    cur_enc_frame = min_encoder_framer - 1
    while (cur_enc_frame < (max_encoder_framer + 1)) and (cur_dec_frame < (max_decoder_framer + 1)):
        cur_enc_frame += 1
        # print()
        # print("--- Поиск соответствий кадра {} ---".format(cur_enc_frame))
        min_mse = 999_999
        min_num = -1
        enc_img = read_img_num(encoder_dir, cur_enc_frame)
        if enc_img is None:
            continue

        search_dec_forward = min(max_decoder_framer, cur_dec_frame + max_dec_forward)

        # Нахождение лучшего соответствия кадру кодера из кадров декодера
        for opposite_dec_frame in range(cur_dec_frame, search_dec_forward + 1):
            if opposite_dec_frame in decoder_frame_cache:
                dec_img = decoder_frame_cache[opposite_dec_frame]
            else:
                dec_img = read_img_num(decoder_dir, opposite_dec_frame)
                decoder_frame_cache[opposite_dec_frame] = dec_img
            if dec_img is None:
                continue
            this_mse = round(mse_metric(enc_img, dec_img), 6)
            if this_mse <= min_mse:
                min_mse = this_mse
                min_num = opposite_dec_frame
        # Обработка случая полного отсутствия соответствий
        if min_num == -1:
            # print("Соответствий не найдено!")
            continue

        # Обработка номеров декодера
        old_dec_frame = cur_dec_frame
        new_dec_frame = min_num
        dec_img = read_img_num(decoder_dir, new_dec_frame)

        min_enc_num = cur_enc_frame  # min_mse
        # Нахождение лучшего соответствия из кадров кодера к текущему кадру декодера
        search_enc_forward = min(max_encoder_framer, cur_enc_frame + max_enc_forward)
        for opposite_enc_frame in range(cur_enc_frame + 1, search_enc_forward + 1):
            if opposite_enc_frame in encoder_frame_cache:
                enc_img = encoder_frame_cache[opposite_enc_frame]
            else:
                enc_img = read_img_num(encoder_dir, opposite_enc_frame)
                encoder_frame_cache[opposite_enc_frame] = enc_img
            this_mse = round(mse_metric(enc_img, dec_img), 6)
            if this_mse <= min_mse:
                min_mse = this_mse
                min_enc_num = opposite_enc_frame

        # Обработка номеров кодера
        old_enc_frame = cur_enc_frame
        new_enc_frame = min_enc_num
        enc_img = read_img_num(encoder_dir, new_enc_frame)

        # Получение метрик
        association_tuple = (cur_enc_frame, min_mse)
        psnr_met = round(psnr_metric(enc_img, dec_img), 6)
        ssim_met = round(ssim_metric(enc_img, dec_img), 6)
        mse_met = min_mse
        lpips_vgg_met, lpips_alex_met = get_lpips_value(enc_img, dec_img)
        lpips_vgg_met = round(lpips_vgg_met, 6)
        lpips_alex_met = round(lpips_alex_met, 6)

        # Обработка случая мнимого соответствия
        if mse_met > mse_threshold:
            # print("Соответствий не найдено!")
            continue

        # Заполнение таблиц
        association_table.append(association_tuple)
        psnr_table.append(psnr_met)
        mse_table.append(mse_met)
        ssim_table.append(ssim_met)
        lpips_vgg_table.append(lpips_vgg_met)
        lpips_alex_table.append(lpips_alex_met)

        # Обработка кеша декодера
        cur_dec_frame = new_dec_frame + 1
        for del_ind in range(old_dec_frame, cur_dec_frame):
            decoder_frame_cache.pop(del_ind, None)

        # Обработка кеша кодера
        cur_enc_frame = new_enc_frame
        for del_ind in range(old_enc_frame, cur_enc_frame + 1):
            encoder_frame_cache.pop(del_ind, None)

        # Вывод данных
        # enc_file = os.path.join(encoder_dir, framer.format(cur_enc_frame))
        # dec_file = os.path.join(decoder_dir, framer.format(min_num))
        # print("Соответствие найдено: {} -> {}!".format(enc_file, dec_file))
        # print("MSE = {:.4f}".format(mse_met))
        # print("PSNR = {:.4f}".format(psnr_met))
        # print("SSIM = {:.4f}".format(ssim_met))

    # mean_mse = round(sum(mse_table) / len(mse_table), 4)
    # mean_psnr = round(sum(psnr_table) / len(psnr_table), 4)
    # mean_ssim = round(sum(ssim_table) / len(ssim_table), 4)
    # loss = round((1 - len(association_table) / enc_frame_num) * 100, 4)

    new_association_table = [str(i[0]) + " " + str(i[1]) for i in association_table]
    result = pd.DataFrame(data={"association_table": new_association_table,
                                "psnr_table": psnr_table,
                                "mse_table": mse_table,
                                "ssim_table": ssim_table,
                                "lpips_vgg_table": lpips_vgg_table,
                                "lpips_alex_table": lpips_alex_table})

    return result


if os.path.isfile(storage_file):
    print("Экспериментальные файлы найдены. Загружаем...")

    dataset = pd.read_pickle(storage_file, compression={'method': 'zip', 'compresslevel': 9})
else:
    print("Экспериментальные файлы не найдены. Проводим эксперименты...")

    import torch
    import lpips

    lpips_alex = lpips.LPIPS(net='alex').to(dtype=torch.float32).cuda()
    lpips_vgg = lpips.LPIPS(net='vgg').to(dtype=torch.float32).cuda()
    dataset = pd.DataFrame(data={"type": [],
                                 "method": [],
                                 "association_table": [],
                                 "psnr_table": [],
                                 "mse_table": [],
                                 "ssim_table": [],
                                 "lpips_vgg_table": [],
                                 "lpips_alex_table": []})
    dataset = dataset.astype({"type": "string", "method": "string",
                              "association_table": "string", "psnr_table": "float64",
                              "mse_table": "float64", "ssim_table": "float64",
                              "lpips_vgg_table": "float64", "lpips_alex_table": "float64"})

    print("=== Стандартные кодеки ===")
    for enc_dir, dec_dir in standard_dirs:
        print("- {} -> {} -".format(enc_dir, dec_dir))
        result = quality_measurer_standard(enc_dir, dec_dir)
        result["type"] = "standard"
        method_name = re.search(method_parser, enc_dir)[1]
        result["method"] = method_name
        res_psnr = result["psnr_table"].to_numpy()
        res_mse = result["mse_table"].to_numpy()
        res_ssim = result["ssim_table"].to_numpy()
        res_lpips_vgg = result["lpips_vgg_table"].to_numpy()
        res_lpips_alex = result["lpips_alex_table"].to_numpy()

        if len(res_psnr) > 0:
            print("PSNRср = {:.04f} дБ".format(res_psnr.mean()))
            print("MSEср = {:.04f}".format(res_mse.mean()))
            print("SSIMср = {:.04f}".format(res_ssim.mean()))
            print("LPIPS[VGG]ср = {:.04f}".format(res_lpips_vgg.mean()))
            print("LPIPS[Alex]ср = {:.04f}".format(res_lpips_alex.mean()))

        dataset = pd.concat([dataset, result], axis=0)
    print("=== Нейросетевые кодеки ===")
    for enc_dir, dec_dir in neuro_dirs:
        print("- {} -> {} -".format(enc_dir, dec_dir))
        result = quality_measurer_neuro(enc_dir, dec_dir)
        result["type"] = "neuro"
        method_name = re.search(method_parser, enc_dir)[1]
        result["method"] = method_name
        res_psnr = result["psnr_table"].to_numpy()
        res_mse = result["mse_table"].to_numpy()
        res_ssim = result["ssim_table"].to_numpy()
        res_lpips_vgg = result["lpips_vgg_table"].to_numpy()
        res_lpips_alex = result["lpips_alex_table"].to_numpy()

        if len(res_psnr) > 0:
            print("PSNRср = {:.04f} дБ".format(res_psnr.mean()))
            print("MSEср = {:.04f}".format(res_mse.mean()))
            print("SSIMср = {:.04f}".format(res_ssim.mean()))
            print("LPIPS[VGG]ср = {:.04f}".format(res_lpips_vgg.mean()))
            print("LPIPS[Alex]ср = {:.04f}".format(res_lpips_alex.mean()))

        dataset = pd.concat([dataset, result], axis=0)

    pd.to_pickle(dataset, storage_file, compression={'method': 'zip', 'compresslevel': 9}, protocol=5)

    print("Экспериментальные файлы сохранены.")

print(dataset)

standard_psnr = dataset[dataset["type"] == "standard"]["psnr_table"].to_numpy()
standard_mse = dataset[dataset["type"] == "standard"]["mse_table"].to_numpy()
standard_ssim = dataset[dataset["type"] == "standard"]["ssim_table"].to_numpy()
standard_lpips_vgg = dataset[dataset["type"] == "standard"]["lpips_vgg_table"].to_numpy()
standard_lpips_alex = dataset[dataset["type"] == "standard"]["lpips_alex_table"].to_numpy()

neuro_psnr = dataset[dataset["type"] == "neuro"]["psnr_table"].to_numpy()
neuro_mse = dataset[dataset["type"] == "neuro"]["mse_table"].to_numpy()
neuro_ssim = dataset[dataset["type"] == "neuro"]["ssim_table"].to_numpy()
neuro_lpips_vgg = dataset[dataset["type"] == "neuro"]["lpips_vgg_table"].to_numpy()
neuro_lpips_alex = dataset[dataset["type"] == "neuro"]["lpips_alex_table"].to_numpy()

all_psnr = dataset["psnr_table"].to_numpy()
all_mse = dataset["mse_table"].to_numpy()
all_ssim = dataset["ssim_table"].to_numpy()
all_lpips_vgg = dataset["lpips_vgg_table"].to_numpy()

plt.xlabel("PSNR (ст.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(standard_psnr, standard_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_psnr_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("MSE (ст.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(standard_mse, standard_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_mse_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("SSIM (ст.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(standard_ssim, standard_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_ssim_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("PSNR (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(standard_psnr, standard_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_psnr_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("MSE (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(standard_mse, standard_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_mse_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("SSIM (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(standard_ssim, standard_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\st_ssim_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("PSNR (нейр.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(neuro_psnr, neuro_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_psnr_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("MSE (ст.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(neuro_mse, neuro_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_mse_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("SSIM (ст.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(neuro_ssim, neuro_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_ssim_lpips_vgg_scatter.jpg")
plt.close()


plt.xlabel("PSNR (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(neuro_psnr, neuro_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_psnr_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("MSE (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(neuro_mse, neuro_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_mse_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("SSIM (ст.), дБ")
plt.ylabel("LPIPS[Alex]")
plt.scatter(neuro_ssim, neuro_lpips_alex)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_ssim_lpips_alex_scatter.jpg")
plt.close()

plt.xlabel("SSIM")
plt.ylabel("LPIPS[VGG]")
plt.scatter(all_ssim, all_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\ssim_lpips_vgg_scatter.jpg")  # !!!
plt.close()

plt.xlabel("PSNR, дБ")
plt.ylabel("LPIPS[VGG]")
plt.scatter(all_psnr, all_lpips_vgg)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\psnr_lpips_vgg_scatter.jpg")
plt.close()

plt.xlabel("PSNR (нейр.), дБ")
plt.ylabel("SSIM")
plt.scatter(neuro_psnr, neuro_ssim)
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\neur_psnr_ssim_scatter.jpg")  # !!!
plt.close()

print("=== Стандартные кодеки: Корреляция ===")
r_st_psnr_lpips_vgg = scipy.stats.pearsonr(standard_psnr, standard_lpips_vgg).statistic
print("r(PSNR, LPIPS[VGG]) = {:.04f}".format(r_st_psnr_lpips_vgg))
r_st_psnr_lpips_alex = scipy.stats.pearsonr(standard_psnr, standard_lpips_alex).statistic
print("r(PSNR, LPIPS[Alex]) = {:.04f}".format(r_st_psnr_lpips_alex))
r_st_mse_lpips_vgg = scipy.stats.pearsonr(standard_mse, standard_lpips_vgg).statistic
print("r(MSE, LPIPS[VGG]) = {:.04f}".format(r_st_mse_lpips_vgg))
r_st_mse_lpips_alex = scipy.stats.pearsonr(standard_mse, standard_lpips_alex).statistic
print("r(MSE, LPIPS[Alex]) = {:.04f}".format(r_st_mse_lpips_alex))
r_st_ssim_lpips_vgg = scipy.stats.pearsonr(standard_ssim, standard_lpips_vgg).statistic
print("r(SSIM, LPIPS[VGG]) = {:.04f}".format(r_st_ssim_lpips_vgg))
r_st_ssim_lpips_alex = scipy.stats.pearsonr(standard_ssim, standard_lpips_alex).statistic
print("r(SSIM, LPIPS[Alex]) = {:.04f}".format(r_st_ssim_lpips_alex))

print("=== Нейросетевые кодеки: Корреляция ===")
r_neur_psnr_lpips_vgg = scipy.stats.pearsonr(neuro_psnr, neuro_lpips_vgg).statistic
print("r(PSNR, LPIPS[VGG]) = {:.04f}".format(r_neur_psnr_lpips_vgg))
r_neur_psnr_lpips_alex = scipy.stats.pearsonr(neuro_psnr, neuro_lpips_alex).statistic
print("r(PSNR, LPIPS[Alex]) = {:.04f}".format(r_neur_psnr_lpips_alex))
r_neur_mse_lpips_vgg = scipy.stats.pearsonr(neuro_mse, neuro_lpips_vgg).statistic
print("r(MSE, LPIPS[VGG]) = {:.04f}".format(r_neur_mse_lpips_vgg))
r_neur_mse_lpips_alex = scipy.stats.pearsonr(neuro_mse, neuro_lpips_alex).statistic
print("r(MSE, LPIPS[Alex]) = {:.04f}".format(r_neur_mse_lpips_alex))
r_neur_ssim_lpips_vgg = scipy.stats.pearsonr(neuro_ssim, neuro_lpips_vgg).statistic
print("r(SSIM, LPIPS[VGG]) = {:.04f}".format(r_neur_ssim_lpips_vgg))
r_neur_ssim_lpips_alex = scipy.stats.pearsonr(neuro_ssim, neuro_lpips_alex).statistic
print("r(SSIM, LPIPS[Alex]) = {:.04f}".format(r_neur_ssim_lpips_alex))

r_ssim_lpips_vgg = scipy.stats.pearsonr(all_ssim, all_lpips_vgg).statistic
print("Общая r(SSIM, LPIPS[VGG]) = {:.04f}".format(r_ssim_lpips_vgg))
r_neur_ssim_psnr = scipy.stats.pearsonr(neuro_ssim, neuro_psnr).statistic
print("Нейро r(PSNR, SSIM) = {:.04f}".format(r_neur_ssim_psnr))
r_st_ssim_psnr = scipy.stats.pearsonr(standard_ssim, standard_psnr).statistic
print("Стандарт r(PSNR, SSIM) = {:.04f}".format(r_st_ssim_psnr))
r_neur_ssim_mse = scipy.stats.pearsonr(neuro_ssim, neuro_mse).statistic
print("Нейро r(SSIM, MSE) = {:.04f}".format(r_neur_ssim_mse))

r_ssim_psnr = scipy.stats.pearsonr(all_ssim, all_psnr).statistic
print("Общая r(SSIM, PSNR) = {:.04f}".format(r_ssim_psnr))
r_ssim_mse = scipy.stats.pearsonr(all_ssim, all_mse).statistic
print("Общая r(SSIM, MSE) = {:.04f}".format(r_ssim_mse))
r_psnr_lpips_vgg = scipy.stats.pearsonr(all_psnr, all_lpips_vgg).statistic
print("Общая r(PSNR, LPIPS[VGG]) = {:.04f}".format(r_psnr_lpips_vgg))


# > 37
# ---
# 31 <= PSNR < 37
first_st_lpips_vgg = dataset[(dataset["type"] == "standard") & (dataset["psnr_table"] >= 31) & (dataset["psnr_table"] < 37)]["lpips_vgg_table"].to_numpy()
# 25 <= PSNR < 31
second_st_lpips_vgg = dataset[(dataset["type"] == "standard") & (dataset["psnr_table"] >= 25) & (dataset["psnr_table"] < 31)]["lpips_vgg_table"].to_numpy()
# 20 <= PSNR < 25
third_st_lpips_vgg = dataset[(dataset["type"] == "standard") & (dataset["psnr_table"] >= 20) & (dataset["psnr_table"] < 25)]["lpips_vgg_table"].to_numpy()
# PSNR < 20
print()
print("=== Попытка найти шкалу ===")
print("При 31 <= PSNR < 37: {:.04f} <= LPIPS < {:.04f}".format(first_st_lpips_vgg.min(), first_st_lpips_vgg.max()))
print("При 25 <= PSNR < 31: {:.04f} <= LPIPS < {:.04f}".format(second_st_lpips_vgg.min(), second_st_lpips_vgg.max()))
# print("При 20 <= PSNR < 25: {:.04f} <= LPIPS < {:.04f}".format(third_st_lpips_vgg.min(), third_st_lpips_vgg.max()))

# Нахождение наименьших квадратов для всех SSIM и LPIPS[VGG]

new_all_ssim = np.append(all_ssim, [0, 1])
new_all_lpips_vgg = np.append(all_lpips_vgg, [1, 0])

# x - all_ssim
# y_true - all_lpips_vgg

# reg1 = lambda ssim, p: (ssim ** 2) * (-p[0]) + p[1]
# [0.65392979 0.78852769]

# reg1 = lambda ssim, p: np.sqrt(ssim * (-p[0]) + p[1])
# [0.7991367  0.78052975]

reg1 = lambda ssim, p: 1 - np.exp(ssim * p[0] - p[1])
# [1.707174   1.82985578]

def res_func(x):
    est_lpips_vgg = reg1(all_ssim, x)
    rem = all_lpips_vgg - est_lpips_vgg
    return rem



print()
print("=== Нахождение наименьших квадратов для всех SSIM и LPIPS[VGG] ===")
res = scipy.optimize.least_squares(res_func, (0.5, 1))
reg1_x = res.x
print(reg1_x)
ssim_space = np.linspace(0, 1, 100)
est_lpips_vgg = reg1(ssim_space, reg1_x)

plt.xlabel("SSIM")
plt.ylabel("LPIPS[VGG]")
plt.scatter(all_ssim, all_lpips_vgg)
plt.plot(ssim_space, est_lpips_vgg, "r-")
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\1.jpg")  # !!!
plt.close()

# Нахождение наименьших квадратов для нейронных PSNR и SSIM

# reg2 = lambda psnr, p: (psnr ** 2) * p[0] + psnr * p[1] + p[2]
# [ -0.06837393   4.22079177 -64.42358082]

reg2 = lambda psnr, p: 1 - np.exp(psnr * (-p[0]) + p[1])
# [ 0.46512274 12.84399696]

# reg2 = lambda psnr, p: np.sqrt(psnr*p[0] - p[1])
# [0.22441791 6.28231265]


def res_func2(x):
    est_ssim = reg2(neuro_psnr, x)
    rem = neuro_ssim - est_ssim
    return rem


print("=== Нахождение наименьших квадратов для нейронных PSNR и SSIM ===")
res2 = scipy.optimize.least_squares(res_func2, (0.5, 18.0))
# res2 = scipy.optimize.least_squares(res_func2, (0.5, 5))
reg2_x = res2.x
print(reg2_x)
psnr_space = np.linspace(neuro_psnr.min(), neuro_psnr.max(), 100)
est_ssim = reg2(psnr_space, reg2_x)

plt.xlabel("PSNR (нейр.), дБ")
plt.ylabel("SSIM")
plt.scatter(neuro_psnr, neuro_ssim)
plt.plot(psnr_space, est_ssim, "r-")
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\2.jpg")  # !!!
plt.close()

print()
print("=== Нахождение значений ===")
all_space = np.linspace(0, 40, 100)
pre_ssim = reg2(all_space, reg2_x)
pre_ssim[pre_ssim > 1.0] = 1.0
pre_ssim[pre_ssim < 0.0] = 0.0
pre_lpips_vgg = reg1(pre_ssim, reg1_x)
pre_lpips_vgg[pre_lpips_vgg > 1.0] = 1.0
pre_lpips_vgg[pre_lpips_vgg < 0.0] = 0.0

plt.xlabel("PSNR (нейр.), дБ")
plt.ylabel("LPIPS[VGG]")
plt.plot(all_space, pre_lpips_vgg, "r")
plt.savefig(r"D:\UserData\Работа\Проекты_статей\НТК_ППС2026\final.jpg")  # !!!
plt.close()

f1 = lambda lpips: ((np.log(1 - lpips) + 1.8299) / 1.7072)
f2 = lambda lpips: (np.log(1 - ((np.log(1 - lpips) + 1.8299) / 1.7072)) - 12.8440) / -0.4651
eps = 0.01
lps = [0.25, 0.5, 0.75]
for lp in lps:
    rs1 = f1(lp)
    rs = f2(lp)
    print("При LPIPS[VGG] = {:.04f}: SSIM = {:.04f}; PSNR = {:.04f}".format(lp, rs1, rs))


