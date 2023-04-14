import numpy as np
from scipy.stats import stats


def cor_pirson (data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    corr = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
    # corr = ((data1 * data2).mean() - mean1 * mean2) / (std1 * std2)
    return corr
