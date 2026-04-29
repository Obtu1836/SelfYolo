import torch as th
from torch import nn


class SPPF(nn.Module):
    def __init__(self,  in_channels: int,
                        radio: float = 0.5,
                        pool_size: int = 5):
        
        super().__init__()

        mid_channels = int(in_channels*radio)
        self.conv1 = self._Conv(in_channels, mid_channels)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size//2)
        self.conv2 = self._Conv(4*mid_channels, in_channels)

    def _Conv(self, ins, ous):

        layer = nn.Sequential(
            nn.Conv2d(ins, ous, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ous),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):

        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.conv2(th.cat([x, y1, y2, y3], dim=1))
    
if __name__ == '__main__':
    data=th.rand(64,512,15,15)

    net=SPPF(512)
    print(net(data).shape)


