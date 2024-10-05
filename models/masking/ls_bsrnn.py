import torch


class LS_BSRNN(torch.nn.Module):
    """Восстановление после маскирования.
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        tensor_2 = self.encoder(x)
        tensor_4 = self.middleware(tensor_2)
        y = self.middleware(tensor_4)

        return y

