import torch


class LS_BSRSPNN(torch.nn.Module):
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


class LS_BSRSFNN(torch.nn.Module):
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

