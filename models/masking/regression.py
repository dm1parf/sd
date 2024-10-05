import math
import scipy
import numpy as np


class MLRPN_LS:
    default_wight_path = "./weights/mlrpn_ls.npz"

    def __init__(self, length, p, bandwidth):
        self.length = length
        self.p = p
        self.bandwidth = bandwidth

        self._dest_l = int(self.length * (1 - self.p))
        self._dest_m = self.length - self._dest_l

        # Index -- mask value
        # Internal list -- bin values
        self._good_indexes = [self.get_upr_lwr(i) for i in range(self._dest_m)]

        self.K = np.zeros((self._dest_l-1, self._dest_m-1))

    def train(self, bn, pmsk):
        """Обучение модели."""

        for v in range(self._dest_m):
            k_init = np.zeros((self.bandwidth + 1,))
            this_pmsk = pmsk[:, v]
            lwr, upr = self._good_indexes[v]
            this_bn = bn[:, lwr:(upr+1)]
            this_bn = self.one_appendix(this_bn)

            def difference(koef):
                diff_vals = []

                for bn_val, pmsk_val in zip(this_bn, this_pmsk):
                    pmsk_appr = koef[0]
                    for i in range(1, len(bn_val)):
                        pmsk_appr += koef[i] * bn_val[i]
                    pmsk_appr = 255 / (1 + math.exp(-pmsk_appr))

                    difference = pmsk_appr - pmsk_val
                    diff_vals.append(difference)

                diff_vals = np.array(diff_vals)
                return diff_vals

            res = scipy.optimize.least_squares(difference, k_init)
            res_k = res.x
            self.K[:, v] = 0
            self.K[0, v] = res_k[0]
            this_inder = lwr
            for i in range(1, len(res_k)):
                self.K[this_inder, v] = res_k[i]
                this_inder += 1

    def eval(self, biner):
        """Использование модели."""

        biner_ = self.one_appendix(biner)
        record_num = biner_.shape[0]
        pmsk = np.zeros((record_num, self._dest_m))

        for r in range(record_num):
            for v in range(self._dest_m):
                pre_pmsk = self.K[0, v]
                lwr, upr = self._good_indexes[v]
                for i in range(lwr, upr+1):
                    pre_pmsk += self.K[i+1, v] * biner[r, i+1]
                pmsk[r, v] = 255 / (1 + math.exp(-pre_pmsk))

        pmsk = np.clip(pmsk, a_min=0, a_max=255)

        return pmsk

    def save(self, filer=default_wight_path):
        """Сохранение весов."""

        np.save(filer, self.K)

    def load(self, filer=default_wight_path):
        """Сохранение весов."""

        self.K = np.load(filer)

    def get_upr_lwr(self, ind):
        """Получить upr и lwr из mask-значения."""

        frac = ind / (self._dest_m - 1)
        center = frac * self._dest_l

        lwr = center - self.bandwidth / 2
        upr = center + self.bandwidth / 2

        if lwr < 0:
            diff = 0 - lwr
            lwr = lwr + diff
            upr = upr + diff
        if upr > (self._dest_l - 1):
            diff = upr - (self._dest_l - 1)
            lwr = lwr - diff
            upr = upr - diff

        if lwr < 0:
            lwr = 0

        lwr = math.ceil(lwr)
        upr = math.floor(upr)

        if (upr - lwr) > (self.bandwidth - 1):
            if frac > 0.5:
                lwr += 1
            else:
                upr -= 1

        return lwr, upr

    @staticmethod
    def one_appendix(arr):

        help_shaper = arr.shape[0]
        help_ones = np.ones((help_shaper, 1))
        arr_ = np.concatenate([help_ones, arr], axis=1)

        return arr_


class LRPN_LS(MLRPN_LS):
    default_wight_path = "./weights/lrpn_ls.npz"

    def __init__(self, length, p, bandwidth):
        super().__init__(length, p, bandwidth)

    def train(self, bn, pmsk):
        """Обучение модели."""

        bn_ = self.one_appendix(bn)

        k1 = np.matmul(bn_.T, bn_)
        k2 = np.matmul(np.linalg.pinv(k1), bn_.T)
        k3 = np.matmul(k2, pmsk)
        self.K = k3

    def eval(self, biner):
        """Использование модели."""

        biner_ = self.one_appendix(biner)
        record_num = biner_.shape[0]
        pmsk = np.zeros((record_num, self._dest_m))

        for r in range(record_num):
            for v in range(self._dest_m):
                pmsk[r, v] = self.K[0, v]
                for i in self._good_indexes[v]:
                    pmsk[r, v] += self.K[i+1, v] * biner[r, i+1]

        pmsk = np.clip(pmsk, a_min=0, a_max=255)

        return pmsk


if __name__ == "__main__":
    p = 0.2
    length = 14

    model = MLRPN_LS(length=length, p=p, bandwidth=5)

    print(model._dest_l)
    print(model._dest_m)
    upr, lwr = model.get_upr_lwr(2)
    print(upr, lwr)
