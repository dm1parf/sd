import os
import re

import pandas

SAVE_OUTPUT = 'output'


def create_dir(target_path: str, new_dir_name: str, index: str = ""):
    os.makedirs(f"{target_path}/{new_dir_name}/")


def main(data_dir, type_file):
    if not os.path.exists(SAVE_OUTPUT):
        os.makedirs(SAVE_OUTPUT)
    data_frame_list = {}
    for dir_video_metrics in os.listdir(data_dir):
        print(dir_video_metrics)

        if type_file == 'txt' or 'txt and xlsx':
            if not os.path.exists(f'{SAVE_OUTPUT}/{dir_video_metrics}'):
                create_dir(SAVE_OUTPUT, dir_video_metrics)

        # Получаем список файлов и папок в директории
        files = os.listdir(f'{data_dir}/{dir_video_metrics}')

        # Извлекаем числа из имен файлов и создаем список кортежей (имя_файла, число)
        file_numbers = []
        for file_name in files:
            match = re.search(r'^(\d+)_run', file_name)
            if match:
                file_number = int(match.group(1))
                file_numbers.append((file_name, file_number))

        # Сортируем список кортежей по числам
        file_numbers.sort(key=lambda x: x[1])

        count = 1
        metrics = {}
        for frame, frame_number in file_numbers:
            for filename in os.listdir(f'{data_dir}/{dir_video_metrics}/{frame}'):
                if filename == 'metrics.txt':
                    f = os.path.join(f'{data_dir}/{dir_video_metrics}/{frame}', filename)
                    f = open(f)
                    metric_lines = f.readlines()

                    # создание словаря для хранения метрик
                    metrics_frame = {}
                    for line in metric_lines:
                        if line.startswith("image_name"):
                            continue
                        if line.startswith("frame_compression_time"):
                            break
                        # разделение строки на название метрики и значение метрики
                        metric_name, metric_value = line.split(' = ')

                        if metric_name == "vmaf":
                            s = line
                            vmaf_regex = r"'vmaf':\s(\d+\.\d+)"
                            vmaf_match = re.search(vmaf_regex, s)

                            if vmaf_match:
                                vmaf = float(vmaf_match.group(1))
                            else:
                                print("No vmaf found")

                            metrics_frame[metric_name] = "{:.15f}".format(vmaf)
                            continue
                        # сохранение значения метрики в словаре metrics
                        metrics_frame[metric_name] = "{:.15f}".format(float(metric_value))
                        # metrics_frame[metric_name] = float(metric_value)
                        print(metrics_frame)

            metrics[f'{frame}_frame'] = metrics_frame

            # extract metric values
            ssim_data = []
            pirson_data = []
            cosine_similarity = []
            mse = []
            hamming_distance = []
            lpips = []
            erqa = []
            y_msssim = []
            y_psnr = []
            y_ssim = []
            lossless = []

            for key in metrics:
                ssim_data.append(metrics[key]['ssim_data'])
                pirson_data.append(metrics[key]['pirson_data'])
                cosine_similarity.append(metrics[key]['cosine_similarity'])
                mse.append(metrics[key]['mse'])
                hamming_distance.append(metrics[key]['hamming_distance'])
                lpips.append(metrics[key]['lpips'])
                erqa.append(metrics[key]['erqa'])
                y_msssim.append(metrics[key]['y_msssim'])
                y_psnr.append(metrics[key]['y_psnr'])
                y_ssim.append(metrics[key]['y_ssim'])
                lossless.append(metrics[key]['lossless_compression'])

            video_metrics = {'ssim_data        ': ssim_data,
                             'pirson_data      ': pirson_data,
                             'cosine_similarity': cosine_similarity,
                             'mse              ': mse,
                             'hamming_distance ': hamming_distance,
                             'lpips            ': lpips,
                             'erqa             ': erqa,
                             'y_msssim         ': y_msssim,
                             'y_psnr           ': y_psnr,
                             'y_ssim           ': y_ssim,
                             'lossless_compression': lossless}

            count = count + 1

            df = pandas.DataFrame(video_metrics)
            if type_file == 'txt' or 'txt and xlsx':
                df.to_csv(f'{SAVE_OUTPUT}/{dir_video_metrics}/metrics.txt', sep='\t')

            if type_file == 'xlsx' or 'txt and xlsx':
                data_frame_list[f'{dir_video_metrics}'] = df

    if type_file == 'xlsx' or 'txt and xlsx':
        with pandas.ExcelWriter(f'{SAVE_OUTPUT}/metrics.xlsx') as writer:
            for sheet_name in data_frame_list.keys():
                data_frame_list[sheet_name].to_excel(writer, sheet_name=sheet_name)

    return 0


if __name__ == "__main__":
    selector = '1'
    while selector == '1':
        data_dir = input('Введите полный путь к диреткории: ')
        while not os.path.exists(data_dir):
            data_dir = input('Директориии не существует, попробуйте снова: ')

        type_int_file = input('1 - для excel таблиц, 2 - для txt таблиц, 3 - txt and excel: ')
        while type_int_file not in ('1', '2', '3'):
            type_int_file = input('Пожалуйста, введите из предложенного выбора. 1 - для excel таблиц, '
                                  '2 - для txt таблиц, 3 - txt and excel: ')

        if type_int_file == '1':
            type_file = 'xlsx'
        elif type_int_file == '2':
            type_file = 'txt'
        elif type_int_file == '3':
            type_file = 'txt and xlsx'

        main(data_dir, type_file)

        selector = input('1 - продолжить работу, any - выход: ')