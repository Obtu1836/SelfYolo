import torch as th
from torch import nn


class Conv_Bn_LeakRelu(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 ksize: int,
                 stride: int = 1,
                 pad: int = 0):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class DarkNet19(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            Conv_Bn_LeakRelu(3, 32, 3, 1, 1),
            nn.MaxPool2d(2)  # (b,32,h//2,w//2)
        )

        self.layer_2 = nn.Sequential(
            Conv_Bn_LeakRelu(32, 64, 3, 1, 1),
            nn.MaxPool2d(2)  # (b,64,h//4,w//4)
        )

        self.layer_3 = nn.Sequential(
            Conv_Bn_LeakRelu(64, 128, 3, 1, 1),
            Conv_Bn_LeakRelu(128, 64, 1),
            Conv_Bn_LeakRelu(64, 128, 3, 1, 1),
            nn.MaxPool2d(2)  # (b,128,h//8,w//8)
        )

        self.layer_4 = nn.Sequential(
            Conv_Bn_LeakRelu(128, 256, 3, 1, 1),
            Conv_Bn_LeakRelu(256, 128, 1),
            Conv_Bn_LeakRelu(128, 256, 3, 1, 1)
        )  # (b,256,h//8,w//8)

        self.max_pool4 = nn.MaxPool2d(2)  # (b,256,h//16,w//16)

        self.layer_5 = nn.Sequential(
            Conv_Bn_LeakRelu(256, 512, 3, 1, 1),
            Conv_Bn_LeakRelu(512, 256, 1),
            Conv_Bn_LeakRelu(256, 512, 3, 1, 1),
            Conv_Bn_LeakRelu(512, 256, 1),
            Conv_Bn_LeakRelu(256, 512, 3, 1, 1)  # (b,512,h//16,w//16)
        )

        self.max_pool5 = nn.MaxPool2d(2)  # (b,512,h//32,w//32)

        self.layer_6 = nn.Sequential(
            Conv_Bn_LeakRelu(512, 1024, 3, 1, 1),
            Conv_Bn_LeakRelu(1024, 512, 1),
            Conv_Bn_LeakRelu(512, 1024, 3, 1, 1),
            Conv_Bn_LeakRelu(1024, 512, 1),
            Conv_Bn_LeakRelu(512, 1024, 3, 1, 1)  # (b,1024,h//32,w//32)
        )

    def forward(self, x):

        c1 = self.layer_1(x)  # 128  2
        c2 = self.layer_2(c1)  # 64  2
        c3 = self.layer_3(c2)  # 128  2
        c3 = self.layer_4(c3)  # 256  1
        c4 = self.layer_5(self.max_pool4(c3))  # 512  2
        c5 = self.layer_6(self.max_pool5(c4))  # 1024  2

        return c5


if __name__ == '__main__':

    Net = DarkNet19()

    data = th.rand(1, 3, 224, 224)

    print(Net(data).shape)
