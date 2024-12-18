import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


source_file = "statistics2.csv"

test_dataset = pd.read_csv(source_file)
test_coder_series = test_dataset["encoding_time"]
test_decoder_series = test_dataset["decoding_time"]
test_coding_time = test_coder_series.to_numpy()[1:]
test_decoding_time = test_decoder_series.to_numpy()[1:]

x_series = np.arange(1, 1 + len(test_coding_time), 1)


time_series = False
simple_mean = True
sliding_mean = True
exp_smooth = True
models_summary = True


if time_series:
    ## Кодер -- тестовый
    print("Кодер -- тестовый")
    plt.xlabel("Номер кадра")
    plt.ylabel("tCD, мс")
    plt.plot(x_series, test_coding_time)
    plt.show()

    ## Декодер -- тестовый
    print("Декодер -- тестовый")
    plt.xlabel("Номер кадра")
    plt.ylabel("tDCD, мс")
    plt.plot(x_series, test_decoding_time)
    plt.show()

encoder_models = []  # (name, MSE, prm)
decoder_models = []  # (name, MSE, prm)

if simple_mean:
    print("Простая средняя: кодер")
    current_mean = 0
    i = 0

    predictions = []
    for t in test_coding_time:
        if i == 0:
            predictions.append(t)
        else:
            predictions.append(current_mean)
        current_mean = (current_mean * i + t) / (i + 1)

        i += 1

    plt.xlabel("Номер кадра")
    plt.ylabel("tCD, мс")
    plt.plot(x_series, test_coding_time, color='blue', label='Реальные значения')
    plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
    plt.legend(loc='upper right')
    plt.show()

    mse = np.mean((test_coding_time - predictions) ** 2)
    model_tuple = ("MEAN", mse)
    encoder_models.append(model_tuple)
    print("MSE", mse)

    print("Простая средняя: декодер")
    predictions = []
    i = 0
    current_mean = 0
    for t in test_decoding_time:
        current_mean = (current_mean * i + t) / (i + 1)
        predictions.append(current_mean)

        i += 1

    plt.xlabel("Номер кадра")
    plt.ylabel("tDCD, мс")
    plt.plot(x_series, test_decoding_time, color='blue', label='Реальные значения')
    plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
    plt.legend(loc='upper right')
    plt.show()

    mse = np.mean((test_decoding_time - predictions) ** 2)
    model_tuple = ("MEAN", mse, "")
    decoder_models.append(model_tuple)
    print("MSE", mse)

if sliding_mean:
    print("Скользящая средняя: кодер")

    for q in range(1, 101, 1):
        current_mean = 0
        i = 0
        predictions = []
        buffer = []
        for t in test_coding_time:
            # q = 3, i = 0 | len(buffer) = 0
            # q = 3, i = 1 | len(buffer) = 1...
            if i == 0:
                predictions.append(t)
            else:
                predictions.append(current_mean)
            if i < q:
                buffer.append(t)
                current_mean = sum(buffer) / len(buffer)
            else:
                deleted_one = buffer.pop(0)
                buffer.append(t)
                current_mean = current_mean + (t - deleted_one) / q

            i += 1

        mse = np.mean((test_coding_time - predictions) ** 2)
        model_tuple = (f"SLMEAN({q})", mse, q)
        encoder_models.append(model_tuple)

        #if q in [2, 5, 10, 25, 50, 100]:
        if q in []:
            print(f"SLMEAN({q}): MSE = {mse}")
            plt.xlabel("Номер кадра")
            plt.ylabel("tCD, мс")
            plt.plot(x_series, test_coding_time, color='blue', label='Реальные значения')
            plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
            plt.legend(loc='upper right')
            plt.show()

    print("- График зависимости MSE от q")
    slmean_models = [i for i in encoder_models if "SLMEAN" in i[0]]
    slmean_q = [i[2] for i in slmean_models]
    slmean_mse = [i[1] for i in slmean_models]
    plt.xlabel("q")
    plt.ylabel("MSE")
    plt.bar(slmean_q, slmean_mse)
    plt.show()

    print("Скользящая средняя: декодер")

    for q in range(1, 101, 1):
        current_mean = 0
        i = 0
        predictions = []
        buffer = []
        for t in test_decoding_time:
            # q = 3, i = 0 | len(buffer) = 0
            # q = 3, i = 1 | len(buffer) = 1...
            if i == 0:
                predictions.append(t)
            else:
                predictions.append(current_mean)
            if i < q:
                buffer.append(t)
                current_mean = sum(buffer) / len(buffer)
            else:
                deleted_one = buffer.pop(0)
                buffer.append(t)
                current_mean = current_mean + (t - deleted_one) / q

            i += 1

        mse = np.mean((test_decoding_time - predictions) ** 2)
        model_tuple = (f"SLMEAN({q})", mse, q)
        decoder_models.append(model_tuple)

        # if q in [2, 5, 10, 25, 50, 100]:
        if q in []:
            print(f"SLMEAN({q}): MSE = {mse}")
            plt.xlabel("Номер кадра")
            plt.ylabel("tDCD, мс")
            plt.plot(x_series, test_decoding_time, color='blue', label='Реальные значения')
            plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
            plt.legend(loc='upper right')
            plt.show()

    print("- График зависимости MSE от q")
    slmean_models = [i for i in decoder_models if "SLMEAN" in i[0]]
    slmean_q = [i[2] for i in slmean_models]
    slmean_mse = [i[1] for i in slmean_models]
    plt.xlabel("q")
    plt.ylabel("MSE")
    plt.bar(slmean_q, slmean_mse)
    plt.show()


if exp_smooth:
    print("Экспоненциальное сглаживание: декодер")

    dest_list = list(np.arange(0.00, 1.01, 0.01))
    for a in dest_list:
        b = 1 - a

        last_value = 0
        last_prediction = 0
        i = 0
        predictions = []
        buffer = []
        for t in test_coding_time:
            # q = 3, i = 0 | len(buffer) = 0
            # q = 3, i = 1 | len(buffer) = 1...
            if i == 0:
                last_value = t
                last_prediction = t
            last_prediction = a * last_value + b * last_prediction
            predictions.append(last_prediction)
            last_value = t

            i += 1

        mse = np.mean((test_coding_time - predictions) ** 2)
        model_tuple = (f"EXP({a})", mse, a)
        encoder_models.append(model_tuple)

        #if a in [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
        if a in []:
            print(f"EXP({a}): MSE = {mse}")
            plt.xlabel("Номер кадра")
            plt.ylabel("tCD, мс")
            plt.plot(x_series, test_coding_time, color='blue', label='Реальные значения')
            plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
            plt.legend(loc='upper right')
            plt.show()

    print("- График зависимости MSE от a")
    slmean_models = [i for i in encoder_models if "EXP" in i[0]]
    slmean_a = [i[2] for i in slmean_models]
    slmean_mse = [i[1] for i in slmean_models]
    # print(*zip(slmean_a, slmean_mse), sep='\n')
    plt.xlabel("a")
    plt.ylabel("MSE")
    plt.xlim([0.00, 1.00])
    plt.plot(slmean_a, slmean_mse)
    plt.show()

    print("Экспоненциальное сглаживание: декодер")

    dest_list = list(np.arange(0.00, 1.01, 0.01))
    for a in dest_list:
        b = 1 - a

        last_value = 0
        last_prediction = 0
        i = 0
        predictions = []
        buffer = []
        for t in test_decoding_time:
            # q = 3, i = 0 | len(buffer) = 0
            # q = 3, i = 1 | len(buffer) = 1...
            if i == 0:
                last_value = t
                last_prediction = t
            last_prediction = a * last_value + b * last_prediction
            predictions.append(last_prediction)
            last_value = t

            i += 1

        mse = np.mean((test_decoding_time - predictions) ** 2)
        model_tuple = (f"EXP({a})", mse, a)
        decoder_models.append(model_tuple)

        #if a in [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
        if a in []:
            print(f"EXP({a}): MSE = {mse}")
            plt.xlabel("Номер кадра")
            plt.ylabel("tDCD, мс")
            plt.plot(x_series, test_decoding_time, color='blue', label='Реальные значения')
            plt.plot(x_series, predictions, color='red', label='Предсказанные значения')
            plt.legend(loc='upper right')
            plt.show()

    print("- График зависимости MSE от a")
    slmean_models = [i for i in decoder_models if "EXP" in i[0]]
    slmean_a = [i[2] for i in slmean_models]
    slmean_mse = [i[1] for i in slmean_models]
    # print(*zip(slmean_a, slmean_mse), sep='\n')
    plt.xlabel("a")
    plt.ylabel("MSE")
    plt.xlim([0.00, 1.00])
    plt.plot(slmean_a, slmean_mse)
    plt.show()

if models_summary:
    print("=== Модели времени кодирования ===")
    encoder_models.sort(key=lambda x: x[1])
    print(*encoder_models, sep='\n')

    print("\n\n\n=== Модели времени декодирования ===")
    decoder_models.sort(key=lambda x: x[1])
    print(*decoder_models, sep='\n')

