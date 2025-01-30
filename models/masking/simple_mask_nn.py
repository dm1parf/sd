import torch


class SBSRNN_M(torch.nn.Module):
    """Восстановление после маскирования: ПНСВБП-М.
    Simple binary sequence restoration neural network -- masked -- SBSRNN-M.
    См. "Маскирование и восстановление бинарного представления
    латентного пространства вариационного автокодировщика KL-f16"."""

    def __init__(self, length, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.length = length
        self.p = p

        self._dest_l = int(self.length * (1 - self.p))
        self._dest_m = self.length - self._dest_l

        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._dest_l,
                            out_features=self.length),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.length,
                            out_features=self.length),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.length,
                            out_features=self._dest_m),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sequence(x)

        return y


class SBSRNN_F(torch.nn.Module):
    """Восстановление после маскирования: ПНСВБП-П.
    Simple binary sequence restoration neural network -- masked -- SBSRNN-M.
    См. "Маскирование и восстановление бинарного представления
    латентного пространства вариационного автокодировщика KL-f16"."""

    def __init__(self, length, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.length = length
        self.p = p

        self._dest_l = int(self.length * (1 - self.p))
        self._len10 = self.length + self._dest_l

        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._dest_l,
                            out_features=self._len10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self._len10,
                            out_features=self._len10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self._len10,
                            out_features=self.length),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sequence(x)

        return y


def test_m():
    print("MASKED TEST")
    length = 16_384
    p = 0.05
    batch_size = 2
    device = "cpu"
    typer = torch.float32
    model = SBSRNN_M(length=length, p=p)

    # model = model.cuda()
    # model = model.type(torch.float16)
    model = model.to(dtype=typer, device=device)
    model.train()

    dest_l = int(length * (1 - p))
    dest_m = length - dest_l
    # dm = length / dest_m
    dm = length / dest_l

    mindex = []
    firster = 0.0
    # i = (length - 1) - (dest_m - 1)*dm / 2
    i = dm
    k = 0
    while k < length:
        if k >= length:
            break
        if k >= i:
            mindex.append(k)
            i += dm
        k += 1
    z = 0
    while len(mindex) < dest_l:
        if z not in mindex:
            mindex.append(z)
        z += 1

    all_latent = torch.rand(size=[batch_size, length], device=device, dtype=typer)
    latent = all_latent[:, mindex].reshape(batch_size, -1)
    print("1:", all_latent.shape[-1], latent.shape[-1], dest_l)

    print(latent.shape)
    mask = model(latent)
    print(mask.shape)
    print("2:", mask.shape[-1], dest_m)

    all_indexes = range(length)
    not_mindex = list(set(all_indexes) - set(mindex))
    restore = torch.zeros(size=[batch_size, length])
    restore[:, not_mindex] = mask[:]
    restore[:, mindex] = latent[:]
    print(restore.shape)


def test_f():
    print("FULL TEST")
    length = 16_384
    p = 0.05
    batch_size = 2
    device = "cpu"
    typer = torch.float32
    model = SBSRNN_F(length=length, p=p)

    # model = model.cuda()
    # model = model.type(torch.float16)
    model = model.to(dtype=typer, device=device)
    model.train()

    dest_l = int(length * (1 - p))
    dest_m = length - dest_l
    # dm = length / dest_m
    dm = length / dest_l

    mindex = []
    firster = 0.0
    # i = (length - 1) - (dest_m - 1)*dm / 2
    i = dm
    k = 0
    while k < length:
        if k >= length:
            break
        if k >= i:
            mindex.append(k)
            i += dm
        k += 1
    z = 0
    while len(mindex) < dest_l:
        if z not in mindex:
            mindex.append(z)
        z += 1

    all_latent = torch.rand(size=[batch_size, length], device=device, dtype=typer)
    latent = all_latent[:, mindex].reshape(batch_size, -1)
    print("1:", all_latent.shape[-1], latent.shape[-1], dest_l)

    print(latent.shape)
    restore = model(latent)
    print(restore.shape)
    print("2:", restore.shape[-1], length)


if __name__ == "__main__":
    test_m()
    test_f()
