import torch as th
from torch import nn
from torch.nn import functional as f


class Block18_34(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       stride: int = 1,
                       shortcut: None | nn.Module = None):
        super().__init__()

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.shortcut = shortcut

    def forward(self, x):

        out = self.left(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out = out+residual
        return f.relu(out, inplace=True)


class Resnet(nn.Module):
    def __init__(self, num_blocks: list[int] = [2, 2, 2, 2]):
        super().__init__()

        self.dims = 64
        self.layer = nn.Sequential(
            nn.Conv2d(3, self.dims, 7, 2, 3, bias=False),
            nn.BatchNorm2d(self.dims),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        layers = []
        for i, num in enumerate(num_blocks):
            if i == 0:
                layers.append(self._make_layer(self.dims, self.dims, num, 1))
            else:
                layers.append(self._make_layer(self.dims, 2*self.dims, num, 2))
                self.dims *= 2

        self.layers = nn.Sequential(*layers)

    def _make_layer(self, in_channels: int, out_channels: int,
                    nums: int, stride: int):

        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = None

        layers = []
        layers.append(Block18_34(in_channels, out_channels, stride, shortcut))
        for i in range(1, nums):
            layers.append(Block18_34(out_channels, out_channels))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.layer(x)
        x = self.layers(x)
        return x


if __name__ == '__main__':

    device='cuda'
    data = th.rand(64, 3, 480, 480).to(device)
    net = Resnet([2, 2, 2, 2])
    net.to(device)
    print(net(data).shape)
    print(net.dims)
