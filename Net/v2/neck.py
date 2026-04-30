import torch as th
from torch import nn
from .backbone import Conv_Bn_LeakRelu


class Neck(nn.Module):
    def __init__(self,  in_channels: int,
                        out_channels: int,
                        pool_size: int,
                        radio: float = 0.5):
        super().__init__()

        mid = int(in_channels*radio)
        self.conv1 = Conv_Bn_LeakRelu(in_channels, mid, 1, 1, 0)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size//2)
        self.conv2 = Conv_Bn_LeakRelu(in_channels*2, out_channels, 1, 1, 0)

    def forward(self, x):

        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.conv2(th.cat([x, y1, y2, y3], dim=1))


if __name__ == '__main__':
    neck = Neck(1024, 512, 5)

    data = th.rand(3, 1024, 7, 7)
    print(neck(data).shape)
