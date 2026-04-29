import torch as th
from torch import nn
from torch.nn import functional as f

from .backbone import Resnet
from .head import Head
from .neck import SPPF

from tools.NMS import multi_cls_nms
from config.v1 import net_param, NetParam


class Yolo(nn.Module):
    def __init__(self,
                 cfg: NetParam,
                 device: str,
                 conf_thresh: float,
                 nms_thresh: float,
                 is_train: bool = True):

        super().__init__()

        self.device = device
        self.stride = cfg.stride
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_class = cfg.num_class
        self.is_train = is_train

        # 构建3主体网络 backbone,neck,head
        self.backbone = Resnet(cfg.block_nums)
        in_dim = self.backbone.dims
        self.neck = SPPF(in_dim)
        self.head = Head(in_dim,
                         num_class=cfg.num_class,
                         out_channels=in_dim,
                         num_cls_head=cfg.num_cls_head,
                         num_reg_head=cfg.num_reg_head)

        # 构建预测层 通过1*1卷积层 降低到需要的维度
        self.pred_obj = nn.Conv2d(in_dim, 1, 1, 1, 0)
        self.pred_cls = nn.Conv2d(in_dim, cfg.num_class, 1, 1, 0)
        self.pred_bbox = nn.Conv2d(in_dim, 4, 1, 1, 0)

    def forward(self, x):
        if not self.is_train:
            return self.interface(x)
        else:
            x = self.backbone(x)
            x = self.neck(x)
            cls_feat, reg_feat = self.head(x)

            pred_obj = self.pred_obj(cls_feat)  # (b,1,h,w)
            pred_cls = self.pred_cls(cls_feat)  # (b,num_class,h,w)
            pred_bboxes = self.pred_bbox(reg_feat)  # (b,4,h,w) (cx,cy,w,h)坐标格式

            self.fmp_size = pred_obj.shape[-2:]

            # 交换维度 m=h*w
            pred_obj = th.permute(pred_obj, (0, 2, 3, 1)
                                  ).contiguous().flatten(1, 2)  # (b,m,1)
            pred_cls = th.permute(pred_cls, (0, 2, 3, 1)).contiguous().flatten(
                1, 2)  # (b,m,num_class)
            pred_bboxes = th.permute(
                # (b,m,4)
                pred_bboxes, (0, 2, 3, 1)).contiguous().flatten(1, 2)

            pred_bboxes = self.decode_bboxes(pred_bboxes, self.fmp_size)
            outputs = {'pred_obj': pred_obj,
                       'pred_cls': pred_cls,
                       'pred_bboxes': pred_bboxes,
                       'stride': self.stride,
                       'fmp_size': self.fmp_size}

            return outputs

    def decode_bboxes(self, bboxes: th.Tensor, fmp_size: tuple[int, int]):
        '''
        生成网格 bboxes每个框格式设定为(cx,cy,w,h) cx,cy为相对当前网格的偏移量
        通过sigmoid 将偏移量映射到(0,1)范围
        w,h 的值要保证非负的实数 且 定义域为全部实数 通过exp实现
        最后通过乘步长 映射回预处理后的图片位置

        由于偏移量 置信度都是处于(0,1)范围内 意味着是基于特征图网格
        w,h是无限制的正数 则是基于真实标注框的信息 

        以上信息需和matcher.py中的逻辑对应
        '''

        grid = self._create_grid(fmp_size)  # 生成网格

        pred_center = (th.sigmoid(bboxes[..., :2])+grid)*self.stride  # (b,M,2)
        pred_wh = th.exp(bboxes[..., 2:])*self.stride  # (b,M,2)
        x1y1 = pred_center-0.5*pred_wh  # (b,M,2)
        x2y2 = pred_center+0.5*pred_wh  # (b,M,2)

        pred_bbox = th.cat([x1y1, x2y2], dim=-1)  # (b,M,4)

        return pred_bbox

    def _create_grid(self, fmp_size: tuple[int, int]):
        '''生成网格坐标 返回形状为(h*w,2)'''
        h, w = fmp_size
        gy, gx = th.meshgrid(th.arange(h), th.arange(w), indexing='ij')
        grid = th.stack((gx, gy), dim=-1).float().view(-1, 2)
        grid = grid.to(self.device)

        return grid

    @th.no_grad()
    def interface(self, x):
        '''
        为简化 推理时 batch=1
        '''
        x = self.backbone(x)
        x = self.neck(x)
        cls_feat, reg_feat = self.head(x)

        pred_obj = self.pred_obj(cls_feat)  # (b,1,h,w)
        pred_cls = self.pred_cls(cls_feat)  # (b,num_class,h,w)
        pred_bboxes = self.pred_bbox(reg_feat)  # (b,4,h,w) (cx,cy,w,h)坐标格式

        self.fmp_size = pred_obj.shape[-2:]

        # 交换维度 m=h*w
        pred_obj = th.permute(pred_obj, (0, 2, 3, 1)
                              ).contiguous().flatten(1, 2)  # (b,m,1)
        pred_cls = th.permute(pred_cls, (0, 2, 3, 1)).contiguous().flatten(
            1, 2)  # (b,m,num_class)
        pred_bboxes = th.permute(
            pred_bboxes, (0, 2, 3, 1)).contiguous().flatten(1, 2)  # (b,m,4)

        pred_obj = pred_obj[0]  # (M,1)
        pred_cls = pred_cls[0]  # (M,num_class)
        pred_bboxes = pred_bboxes[0]  # (M,4)

        # 计算分数
        scores = th.sqrt(pred_obj.sigmoid() *
                         pred_cls.sigmoid())  # (m,num_class)
        bboxes = self.decode_bboxes(
            bboxes=pred_bboxes, fmp_size=self.fmp_size)  # (m,4)

        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # 多类别非极大值抑制
        bboxes, confs, labels = multi_cls_nms(
            bboxes, scores, self.conf_thresh, self.nms_thresh, self.num_class)

        return bboxes, confs, labels


def build_yolo(cfg, device, conf_thresh, nms_thresh, is_train):
    model = Yolo(cfg, device, conf_thresh, nms_thresh, is_train)
    return model


if __name__ == '__main__':

    netparam = net_param

    device = 'mps'

    data = th.rand(64, 3, 480, 480).to(device)
    model = build_yolo(netparam, device, 0.005, 0.2, False)
    model.to(device)
    res = model(data)
    # print(model)
    print(res)
