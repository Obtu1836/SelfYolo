import torch as th
from torch import nn
from .backbone import Conv_Bn_LeakRelu
from config.v2 import NetParam, net_param


class Head(nn.Module):
    def __init__(self, cfg: NetParam,
                 in_channels: int,
                 out_channels: int,
                 num_class: int = 20):
        super().__init__()

        self.in_channesl = in_channels
        self.num_class = num_class

        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head

        cls_feat = []
        cls_channels = max(num_class, out_channels)
        for i in range(self.num_cls_head):
            if i == 0:
                cls_feat.append(Conv_Bn_LeakRelu(
                    in_channels, cls_channels, 3, 1, 1))
            else:
                cls_feat.append(Conv_Bn_LeakRelu(
                    cls_channels, cls_channels, 3, 1, 1))

        reg_feat = []
        reg_channels = max(out_channels, 64)
        for i in range(self.num_reg_head):
            if i == 0:
                reg_feat.append(Conv_Bn_LeakRelu(
                    in_channels, reg_channels, 3, 1, 1))

            else:
                reg_feat.append(Conv_Bn_LeakRelu(
                    reg_channels, reg_channels, 3, 1, 1))

        self.cls_feat = nn.Sequential(*cls_feat)
        self.reg_feat = nn.Sequential(*reg_feat)

    def forward(self, x):

        pred_cls = self.cls_feat(x)
        pred_reg = self.reg_feat(x)

        return pred_cls, pred_reg


if __name__ == '__main__':
    cfg = net_param
    net = Head(cfg, 512, 256, 20)
    data = th.rand(3, 512, 7, 7)
    a, b = net(data)
    print(a.shape, b.shape)
