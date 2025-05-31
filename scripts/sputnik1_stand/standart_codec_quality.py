import os
import cv2
import argparse
from skimage.metrics import structural_similarity
import numpy as np
import math


framer = "frame_{:04d}.bmp"
encoder_dir = "encoder_h266_hd_25"
decoder_dir = "decoder_h266_hd_hd"
max_dec_forward = 200
# Максимальное количество кадров просмотра вперёд.
# Т.е. есть кадр кодировщика 1. Ищет в 200 кадрах декодировщика лучшее соответствие.
max_enc_forward = 50
# Может так оказаться, что соответствие не идеальное.
# И к выбранному кадру декодера есть лучший кадр кодера на соответствие.
mse_threshold = 80.0  # 90
# Порог для MSE
# Если MSE выше, то с высокой вероятностью то не соответствие, а лишь выброс

parser = argparse.ArgumentParser(prog="Измеритель качества стандартных кодеков", description="Измеряет качество стандартных кодеков без выбросов")
parser.add_argument('-e', '--encoder_dir', dest="encoder_dir", type=str, default=encoder_dir)
parser.add_argument('-d', '--decoder_dir', dest="decoder_dir", type=str, default=decoder_dir)
parser.add_argument('--max_dec_forward', dest="max_dec_forward", type=int, default=max_dec_forward)
parser.add_argument('--max_enc_forward', dest="max_enc_forward", type=int, default=max_enc_forward)
parser.add_argument('--mse_threshold', dest="mse_threshold", type=float, default=mse_threshold)
args = parser.parse_args()

encoder_dir = args.encoder_dir
decoder_dir = args.decoder_dir
max_dec_forward = args.max_dec_forward
max_enc_forward = args.max_enc_forward
mse_threshold = args.mse_threshold


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

decoder_frame_cache = dict()
encoder_frame_cache = dict()
cur_dec_frame = min_decoder_framer
# for cur_enc_frame in range(min_encoder_framer, max_encoder_framer + 1):
cur_enc_frame = min_encoder_framer - 1
while cur_enc_frame < (max_encoder_framer + 1):
    cur_enc_frame += 1
    print()
    print("--- Поиск соответствий кадра {} ---".format(cur_enc_frame))
    min_mse = 999_999
    min_num = -1
    enc_img = read_img_num(encoder_dir, cur_enc_frame)

    search_dec_forward = min(max_decoder_framer, cur_dec_frame + max_dec_forward)

    # Нахождение лучшего соответствия кадру кодера из кадров декодера
    for opposite_dec_frame in range(cur_dec_frame, search_dec_forward + 1):
        if opposite_dec_frame in decoder_frame_cache:
            dec_img = decoder_frame_cache[opposite_dec_frame]
        else:
            dec_img = read_img_num(decoder_dir, opposite_dec_frame)
            decoder_frame_cache[opposite_dec_frame] = dec_img
        this_mse = round(mse_metric(enc_img, dec_img), 6)
        if this_mse <= min_mse:
            min_mse = this_mse
            min_num = opposite_dec_frame
    # Обработка случая полного отсутствия соответствий
    if min_num == -1:
        print("Соответствий не найдено!")
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

    # Обработка случая мнимого соответствия
    if mse_met > mse_threshold:
        print("Соответствий не найдено!")
        continue

    # Заполнение таблиц
    association_table.append(association_tuple)
    psnr_table.append(psnr_met)
    mse_table.append(mse_met)
    ssim_table.append(ssim_met)

    # Обработка кеша декодера
    cur_dec_frame = new_dec_frame + 1
    for del_ind in range(old_dec_frame, cur_dec_frame):
        decoder_frame_cache.pop(del_ind, None)

    # Обработка кеша кодера
    cur_enc_frame = new_enc_frame
    for del_ind in range(old_enc_frame, cur_enc_frame + 1):
        encoder_frame_cache.pop(del_ind, None)

    # Вывод данных
    enc_file = os.path.join(encoder_dir, framer.format(cur_enc_frame))
    dec_file = os.path.join(decoder_dir, framer.format(min_num))
    print("Соответствие найдено: {} -> {}!".format(enc_file, dec_file))
    print("MSE = {:.4f}".format(mse_met))
    print("PSNR = {:.4f}".format(psnr_met))
    print("SSIM = {:.4f}".format(ssim_met))


mean_mse = round(sum(mse_table) / len(mse_table), 4)
mean_psnr = round(sum(psnr_table) / len(psnr_table), 4)
mean_ssim = round(sum(ssim_table) / len(ssim_table), 4)
loss = round((1 - len(association_table) / enc_frame_num) * 100, 4)


print()
print("====== РЕЗУЛЬТАТ ======")
print("MSEср = {:.4f}".format(mean_mse))
print("PSNRср = {:.4f} дБ".format(mean_psnr))
print("SSIMср = {:.4f}".format(mean_ssim))
print("Потери = {:.2f}%".format(loss))
print()
