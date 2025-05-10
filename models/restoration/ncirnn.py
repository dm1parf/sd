import torch


class NCIRNN(torch.nn.Module):
    """Neural Codec Image Restoration Neural Network.
    Solely from convolution layers in order to make it faster.
    For 512x512"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._conver = torch.nn.Sequential(
            # 1x3x512x512
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=9),
            # 1x32x504x504
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # 1x32x126x126
            torch.nn.Conv2d(in_channels=32,
                            out_channels=128,
                            kernel_size=5),
            # 1x128x122x122
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4),  # 2
            # 1x128x61x61
        )
        """
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=9),  # padding=4
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=17),  # padding=8
            torch.nn.ReLU(),
            # 1x256x100x100"""
        self._unconver = torch.nn.Sequential(
            # 1x128x61x61
            torch.nn.ConvTranspose2d(in_channels=128,
                                     out_channels=32,
                                     kernel_size=5,
                                     stride=4,  # 2
                                     output_padding=3),  # 1
            # 1x32x126x126
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32,
                                     out_channels=3,
                                     kernel_size=9,
                                     stride=2,
                                     output_padding=1),
            torch.nn.ReLU(),
        )
        """
            # 1x256x100x100
            torch.nn.ConvTranspose2d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=17,
                                     stride=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=9,
                                     stride=1),
            torch.nn.ReLU(), """

    def forward(self, x):
        z = self._conver(x)
        y = self._unconver(z)

        return y


def test():
    import time
    model = NCIRNN()
    img = torch.normal(0.0, 1.0, size=(1, 3, 512, 512))
    a = time.time()
    img_out = model(img)
    b = time.time()

    print(img.shape)
    print(img_out.shape)
    print(round((b - a)*1000, 4))


if __name__ == "__main__":
    test()
