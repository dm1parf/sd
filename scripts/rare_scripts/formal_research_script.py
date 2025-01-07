import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import hurst
from sklearn.neighbors import KernelDensity
from scipy.integrate import simpson
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.special import erf


warnings.filterwarnings("ignore")


time_series = False
herst = True
kernel_dens_estim = False
interval = False
erfer = True
# source_file2 = "statistics2.csv"
source_file2 = r"D:\UserData\Работа\Проекты_статей\О_прогнозировании_времени_выполнения\statistics2.csv"

test_dataset = pd.read_csv(source_file2)
test_coder_series = test_dataset["encoding_time"]
test_decoder_series = test_dataset["decoding_time"]
test_coding_time = test_coder_series.to_numpy()[1:]
test_decoding_time = test_decoder_series.to_numpy()[1:]

x_series = np.arange(1, 1 + len(test_coding_time), 1)

if time_series:
    ## Кодер -- тестовый
    print("Кодер -- тестовый")
    plt.xlabel("Номер кадра")
    plt.ylabel("tCD, мс")
    plt.plot(x_series, test_coding_time, color='black')
    plt.show()

    ## Декодер -- тестовый
    print("Декодер -- тестовый")
    plt.xlabel("Номер кадра")
    plt.ylabel("tDCD, мс")
    plt.plot(x_series, test_decoding_time, color='black')
    plt.show()

if herst:
    print("Оценка параметра Хёрста для кодера")
    H, c, data = hurst.compute_Hc(test_coding_time, kind='change', simplified=False)
    print(H)
    plt.plot(data[0], c * data[0] ** H, color="grey")
    plt.scatter(data[0], data[1], color="black")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Номер кадра')
    plt.ylabel('R/S')
    plt.grid(True)
    plt.show()

    print("Оценка параметра Хёрста для декодера")
    H, c, data = hurst.compute_Hc(test_decoding_time, kind='change', simplified=False)
    print(H)
    plt.plot(data[0], c * data[0] ** H, color="grey")
    plt.scatter(data[0], data[1], color="black")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Номер кадра')
    plt.ylabel('R/S')
    plt.grid(True)
    plt.show()


if kernel_dens_estim:
    band = 0.03
    # silverman 2.5
    # kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
    kde = KernelDensity(kernel="gaussian", bandwidth=band)
    very_dataset = test_coding_time
    very_dataset = very_dataset.reshape(-1, 1)

    kde = kde.fit(very_dataset)

    step = 0.01
    check = np.arange(very_dataset.min(), very_dataset.max() + step, step)
    check_ = check.reshape(-1, 1)
    y_data = np.exp(kde.score_samples(check_))# * 100

    z = simpson(y_data, x=check)
    print("=== Время кодирования ===")
    print("Check:", z)
    print("Mean:", test_coding_time.mean())
    print("Std:", test_coding_time.std())
    print("Var:", test_coding_time.var())

    plt.xlabel("tCD, мс")
    plt.ylabel("Плотность вероятности")
    plt.plot(check, y_data, color='black')
    plt.show()

    band = 0.03
    # silverman 2.5
    # kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
    kde = KernelDensity(kernel="gaussian", bandwidth=band)
    very_dataset = test_decoding_time
    very_dataset = very_dataset.reshape(-1, 1)

    kde = kde.fit(very_dataset)

    step = 0.01
    check = np.arange(very_dataset.min(), very_dataset.max() + step, step)
    check_ = check.reshape(-1, 1)
    y_data = np.exp(kde.score_samples(check_))# * 100

    z = simpson(y_data, x=check)
    print("=== Время декодирования ===")
    print("Check:", z)
    print("Mean:", test_decoding_time.mean())
    print("Std:", test_decoding_time.std())
    print("Var:", test_decoding_time.var())

    plt.xlabel("tDCD, мс")
    plt.ylabel("Плотность вероятности")
    plt.xlim([test_decoding_time.min(), 49])
    plt.plot(check, y_data, color='black')
    plt.show()

if interval:
    print("=== Время кодирования ===")
    enc_int = st.t.interval(0.95, len(test_coding_time)-1, loc=test_coding_time.mean(), scale=st.sem(test_coding_time))
    enc_int2 = sms.DescrStatsW(test_coding_time).tconfint_mean(alpha=0.05)
    print(enc_int)
    print(enc_int2)

    print("=== Время декодирования ===")
    dec_int = st.t.interval(0.95, len(test_decoding_time)-1, loc=test_decoding_time.mean(), scale=st.sem(test_decoding_time))
    dec_int2 = sms.DescrStatsW(test_decoding_time).tconfint_mean(alpha=0.05)
    print(dec_int)
    print(dec_int2)

if erfer:
    std_z = 5
    mean_z = np.arange(66, 100, 0.1)
    y_func = lambda x: 1/2 * (erf((77.8433 + 2*x)/(np.sqrt(0.0812 + 8 * std_z**2))) - erf((2*x - 172.1567)/(np.sqrt(0.0812 + 8 * std_z**2))))

    k_p = y_func(mean_z)

    plt.xlabel("Tз, мс")
    plt.ylabel("Kp")
    plt.plot(mean_z, k_p, color='black')
    plt.show()

