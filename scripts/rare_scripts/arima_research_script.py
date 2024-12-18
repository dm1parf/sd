import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")


lag_num = 25
time_series = True
adf_test = False
correlation = False
arima_build = False
this_k = 0
p_range = range(0, 11, 1)
q_range = range(0, 11, 1)
source_file1 = "statistics1.csv"
source_file2 = "statistics2.csv"

train_dataset = pd.read_csv(source_file1)
train_coder_series = train_dataset["encoding_time"]
train_decoder_series = train_dataset["decoding_time"]
train_coding_time = train_coder_series.to_numpy()[1:]
train_decoding_time = train_decoder_series.to_numpy()[1:]

test_dataset = pd.read_csv(source_file2)
test_coder_series = test_dataset["encoding_time"]
test_decoder_series = test_dataset["decoding_time"]
test_coding_time = test_coder_series.to_numpy()[1:]
test_decoding_time = test_decoder_series.to_numpy()[1:]

x_series = np.arange(1, 1 + len(train_coding_time), 1)

if time_series:
    # Временные ряды
    ## Кодер -- тренировочный
    print("Кодер -- тренировочный")
    plt.xlabel("Номер кадра")
    plt.ylabel("tCD, мс")
    plt.plot(x_series, train_coding_time)
    plt.show()

    ## Декодер -- тренировочный
    print("Декодер -- тренировочный")
    plt.xlabel("Номер кадра")
    plt.ylabel("tDCD, мс")
    plt.plot(x_series, train_decoding_time)
    plt.show()

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

if adf_test:
    # ADF-тест
    ## Кодер -- тренировочный
    print("Кодер -- тренировочный")
    print(sm.tsa.stattools.adfuller(train_coding_time))

    ## Декодер -- тренировочный
    print("Декодер -- тренировочный")
    print(sm.tsa.stattools.adfuller(train_decoding_time))

    ## Кодер -- тестовый
    print("Кодер -- тестовый")
    print(sm.tsa.stattools.adfuller(test_coding_time))

    ## Декодер -- тестовый
    print("Декодер -- тестовый")
    print(sm.tsa.stattools.adfuller(test_decoding_time))

if correlation:
    # Корреляции
    ## Кодирование - ACF
    print("Кодирование - ACF")
    sm.graphics.tsa.plot_acf(train_coder_series, lags=lag_num)
    plt.title("")
    plt.xlabel("Лаг")
    plt.ylabel("ACF")
    plt.show()

    ## Кодирование - PACF
    print("Кодирование - PACF")
    sm.graphics.tsa.plot_pacf(train_coder_series, lags=lag_num)
    plt.title("")
    plt.xlabel("Лаг")
    plt.ylabel("PACF")
    plt.show()

    ## Декодирование - ACF
    print("Декодирование - ACF")
    sm.graphics.tsa.plot_acf(train_decoder_series, lags=lag_num)
    plt.title("")
    plt.xlabel("Лаг")
    plt.ylabel("ACF")
    plt.show()

    ## Декодирование - PACF
    print("Декодирование - PACF")
    sm.graphics.tsa.plot_pacf(train_decoder_series, lags=lag_num)
    plt.title("")
    plt.xlabel("Лаг")
    plt.ylabel("PACF")
    plt.show()

if arima_build:
    # Кодировщик
    coder_arimas = []  # (name, MSE, AIC, model, prediction)

    print("P range:", p_range)
    print("Q range:", q_range)
    for this_p in p_range:
        for this_q in q_range:
            arima = sm.tsa.arima.ARIMA(train_coding_time, order=(this_p, this_k, this_q))
            arima = arima.fit()
            arima_ = arima

            aic = arima.aic
            name = f"ARIMA({this_p}, {this_k}, {this_q})"

            predictions = []

            for true_val in test_coding_time:
                pred_val = arima_.forecast(1)
                predictions.append(pred_val.item())
                arima_ = arima_.extend([true_val])

            predictions = np.array(predictions)
            mse = np.mean((test_coding_time - predictions) ** 2)

            new_tuple = (name, mse, aic, arima, predictions)
            coder_arimas.append(new_tuple)

    coder_arimas.sort(key=lambda x: x[1])
    print()
    print("=== ARIMA-модели времени кодирования ===:")
    for i in coder_arimas:
        print(i[:4])
    print("=== Параметры лучшей модели кодирования ({}) ===:".format(coder_arimas[0][0]))
    print(coder_arimas[0][3].params)
    plt.xlabel("Номер кадра")
    plt.ylabel("tCD, мс")
    plt.plot(x_series, test_coding_time, color="blue", label="Настоящие значения")
    plt.plot(x_series, coder_arimas[0][4], color="red", label="Предсказанные значения")
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()
    print()
    print()

    # Декодировщик
    decoder_arimas = []  # (name, MSE, AIC, model, prediction)

    print("P range:", p_range)
    print("Q range:", q_range)
    for this_p in p_range:
        for this_q in q_range:
            arima = sm.tsa.arima.ARIMA(train_decoding_time, order=(this_p, this_k, this_q))
            arima = arima.fit()
            arima_ = arima

            aic = arima.aic
            name = f"ARIMA({this_p}, {this_k}, {this_q})"

            predictions = []

            for true_val in test_decoding_time:
                pred_val = arima_.forecast(1)
                predictions.append(pred_val.item())
                arima_ = arima_.extend([true_val])

            predictions = np.array(predictions)
            mse = np.mean((test_decoding_time - predictions) ** 2)

            new_tuple = (name, mse, aic, arima, predictions)
            decoder_arimas.append(new_tuple)

    decoder_arimas.sort(key=lambda x: x[1])
    print()
    print("=== ARIMA-модели времени декодирования ===:")
    for i in decoder_arimas:
        print(i[:4])
    print("=== Параметры лучшей модели декодирования ({}) ===:".format(decoder_arimas[0][0]))
    print(decoder_arimas[0][3].params)
    plt.xlabel("Номер кадра")
    plt.ylabel("tDCD, мс")
    plt.plot(x_series, test_decoding_time, color="blue", label="Настоящие значения")
    plt.plot(x_series, decoder_arimas[0][4], color="red", label="Предсказанные значения")
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()

