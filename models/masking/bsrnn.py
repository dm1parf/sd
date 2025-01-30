import torch


class BSRNN(torch.nn.Module):
    """Восстановление после маскирования: НСВБП.
    Binary sequence restoration neural network -- BSRNN.
    См. "Маскирование и восстановление бинарного представления
    латентного пространства вариационного автокодировщика KL-f16"."""

    def __init__(self, length, p, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.length = length
        self.p = p

        self._dest_l = int(self.length * (1 - self.p))
        self._dest_m = self.length - self._dest_l
        self._len3 = (((self._dest_l - 8) // 4) - 4) // 2
        self._len5 = (((self._dest_m - 8) // 4) - 4) // 2
        self._len4 = self._len3 + self._len5

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=1,
                            kernel_size=9),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
            torch.nn.Conv1d(in_channels=1,
                            out_channels=1,
                            kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.middleware = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._len3,
                            out_features=self._len4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self._len4,
                            out_features=self._len5),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=5,
                                     stride=2,
                                     output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=9,
                                     stride=4,
                                     output_padding=3),
            torch.nn.ReLU(),
        )
        self.last_fix = torch.nn.Linear(in_features=self._len5 * 8 + 24,
                                        out_features=self._dest_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, -1)
        tensor_2 = self.encoder(x)
        tensor_4 = self.middleware(tensor_2)
        y = self.decoder(tensor_4)
        y = y.reshape(batch_size, -1)
        y = self.last_fix(y)
        return y


def test():
    length = 16_384
    p = 0.05
    batch_size = 2
    device = "cpu"
    typer = torch.float32
    model = BSRNN(length=length, p=p)

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


if __name__ == "__main__":
    test()
