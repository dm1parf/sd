import torch
import numpy as np

class MaskMaster:
    def __init__(self, length, p):
        self.length = length
        self.p = p
        self.dest_l = int(self.length * (1 - self.p))
        self.dest_m = self.length - self.dest_l

        self._aindex = None
        self._mindex = None
        self._lindex = None

    def get_all_index(self):
        if not self._aindex:  # Ленивая инициализация
            self._aindex = range(self.length)

        return self._aindex

    def get_mindex(self):
        if not self._mindex:  # Ленивая инициализация
            dm = self.length / self.dest_l

            mindex = []
            i = dm
            k = 0
            while k < self.length:
                if k >= self.length:
                    break
                if k >= i:
                    mindex.append(k)
                    i += dm
                k += 1
            z = 0
            while len(mindex) < self.dest_l:
                if z not in mindex:
                    mindex.append(z)
                z += 1

            self._mindex = mindex

        return self._mindex

    def get_lindex(self):
        if not self._lindex:  # Ленивая инициализация
            aindex = self.get_all_index()
            mindex = self.get_mindex()
            lindex = list(set(aindex) - set(mindex))
            lindex.sort()

            self._lindex = lindex

        return self._lindex

    def get_latent(self, all_latent):
        mindex = self.get_mindex()
        batch_size = all_latent.shape[0]

        latent = all_latent[:, mindex].reshape(batch_size, -1)
        return latent

    def get_mask(self, all_latent):
        lindex = self.get_lindex()
        batch_size = all_latent.shape[0]

        latent = all_latent[:, lindex].reshape(batch_size, -1)
        return latent

    def recompose(self, latent, mask, batch_size=1):
        mindex = self.get_mindex()
        lindex = self.get_lindex()

        typer = latent.dtype
        if isinstance(latent, np.ndarray):
            restore = np.zeros(shape=[batch_size, self.length], dtype=typer)
        else:
            device = latent.device
            restore = torch.zeros(size=[batch_size, self.length], dtype=typer, device=device)
        restore[:, lindex] = mask[:]
        restore[:, mindex] = latent[:]

        return restore

    def prepare_latent(self, all_latent, typer=torch.float16):
        float_latent = all_latent.to(dtype=typer) / 255.0

        return float_latent

    def restore_latent(self, all_latent, typer=torch.uint8):
        int_latent = (all_latent * 255.0)
        int_latent = int_latent.clip(0.0, 255.0)
        int_latent = int_latent.to(dtype=typer)

        return int_latent
