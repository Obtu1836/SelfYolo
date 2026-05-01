import torch as th
from torch import nn

from .backbone import DarkNet19
from .neck import Neck
from .head import Head
from config.v2 import NetParam, net_param
from tools.NMS import mul_cls_nms_label

from ..absnet import YOLO


class Yolo(YOLO):
    def __init__(self,  cfg: NetParam,
                 device: str,
                 conf: float,
                 nms: float,
                 topk: int,
                 is_train: bool):

        super().__init__()

        self.cfg = cfg
        self.device = device
        self.nms = nms
        self.conf = conf
        self.num_class = cfg.num_class
        self.topk = topk
        self.stride = 32
        self.base=416
        self.is_train = is_train

        self.anchor_size = th.as_tensor(cfg.anchor_size).float().view(-1, 2)
        self.num_anchor = self.anchor_size.shape[0]  # k

        self.backbone = DarkNet19()
        self.neck = Neck(1024, 512, 5)
        self.head = Head(cfg, 512, 512, self.num_class)

        self.pred_obj = nn.Conv2d(512, self.num_anchor, 1, 1, 0)  # (b,k,h,w)
        self.pred_cls = nn.Conv2d(
            # (b,n*num_class,h,w)
            512, self.num_anchor*self.num_class, 1, 1, 0)
        self.pred_boxes = nn.Conv2d(
            512, 4*self.num_anchor, 1, 1, 0)  # (b,k*4,h,w)

    def forward(self, x):
        if not self.is_train:
            return self.interface(x)
        else:
            batch = x.shape[0]
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            pred_obj = self.pred_obj(cls_feat)
            pred_cls = self.pred_cls(cls_feat)
            pred_boxes = self.pred_boxes(reg_feat)  # (b,k*4,h,w)
            fmp_size = pred_obj.shape[-2:]  # (h,w)

            anchors = self.make_grid(fmp_size)  # (b*h*w*k,4)
            # (b,k*1,h,w)----------->(b,k*h*w,1)
            # (b,k*num_class,h,w )-->(b,k*h*w,num_class)
            # (b,k*4,h,w)----------->(b,k*h*w,4)
            pred_obj = th.permute(pred_obj, (0, 2, 3, 1)
                                  # (b,k*h*w,1)
                                  ).contiguous().view(batch, -1, 1)
            pred_cls = th.permute(pred_cls, (0, 2, 3, 1)).contiguous().view(
                batch, -1, self.num_class)  # (b,k*h*w,num_class)
            pred_boxes = th.permute(
                # (b,k*h*w,4)
                pred_boxes, (0, 2, 3, 1)).contiguous().view(batch, -1, 4)

            boxes = self.decode_boxes(anchors, pred_boxes)  # (b,k*h*w,4)

            output = {'pred_obj': pred_obj,  # (b,k*h*w,1)
                      'pred_cls': pred_cls,  # (b,k*h*w,num_class)
                      'pred_boxes': boxes,  # (b,k*h*w,4)
                      'stride': self.stride,
                      'fmp_size': fmp_size}  # [h,w]

            return output

    def make_grid(self, fmp_size: tuple[int, int]):
        '''
        在特征图上生成网格并于 聚类生成anchor组合 
        相比于v1 这个版本因为多了anchor 所以在特征图网格中加入anchor的坐标 引入了repeat函数 通过该函数进行
        某一维度上的扩展
        '''
        h, w = fmp_size
        input_h=h*self.stride
        scale=input_h/self.base  #配置文件中的anchor是在416的图像中聚类得到 需要缩放
        gy, gx = th.meshgrid(th.arange(h), th.arange(w), indexing='ij')
        grid = th.stack([gx, gy], dim=-1).float().view(-1, 2)  # (h*w,2)  m=h*w

        # (m,2)->(m,1,2)->(m,k,2)->(m*k,2)=(h*w*k,2)
        grid = grid[:, None, :].repeat(1, self.num_anchor, 1).view(-1, 2)
        grid = grid.to(self.device)

        # (k,2)->(1,k,2)->(m,k,2)将anchor扩展到每个网格
        anchors=self.anchor_size*scale
        anchors = anchors[None, :].repeat(h*w, 1, 1)
        # (m,k,2)->(m*k,2)=(h*w*k,2) 变形
        anchors = anchors.view(-1, 2).to(self.device)

        anchors = th.cat([grid, anchors], dim=-1)  # (h*w*k,4)

        return anchors

    def decode_boxes(self, anchors: th.Tensor, pred_boxes: th.Tensor):
        # 返回x1,y1,x2,y2格式的boxes
        # (b,h*w*k,2)+(h*w*k,2)
        pred_cxy = (th.sigmoid(pred_boxes[..., :2])+anchors[..., :2])*self.stride
        # anchor中的w,h数值是基于输入尺寸的 所以最后不需要*步长· v1是需要*步长的
        pred_wh = th.exp(pred_boxes[..., 2:])*anchors[..., 2:]
        # (cx,cy,w,h)--->(x1,y1,x2,y2)
        pred_x1y1 = pred_cxy-0.5*pred_wh
        pred_x2y2 = pred_cxy+0.5*pred_wh

        pred_boxes = th.cat([pred_x1y1, pred_x2y2], dim=-1)
        return pred_boxes

    @th.no_grad()
    def interface(self, x):

        bs = x.shape[0]
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        pred_obj = self.pred_obj(cls_feat)  # (bs,n,h,w)
        pred_cls = self.pred_cls(cls_feat)  # (bs,n*num_class,h,w)
        pred_boxes = self.pred_boxes(reg_feat)  # (bs,n*4,h,w)
        fmp_size = pred_obj.shape[-2:]

        pred_obj = th.permute(pred_obj, (0, 2, 3, 1)).contiguous().view(
            bs, -1, 1)  # (bs,h*w*k,1)
        pred_cls = th.permute(pred_cls, (0, 2, 3, 1)).contiguous().view(
            bs, -1, self.num_class)  # (bs,h*w*k,num_class)
        pred_boxes = th.permute(pred_boxes, (0, 2, 3, 1)).contiguous().view(
            bs, -1, 4)  # (bs,h*w*k,4)

        pred_obj = pred_obj[0]
        pred_cls = pred_cls[0]
        pred_boxes = pred_boxes[0]

        anchors = self.make_grid(fmp_size)  # (h*w*k,4)
        boxes, scores, labels = self.postprocess(
            pred_obj, pred_cls, pred_boxes, anchors)
        return boxes, scores, labels

    def postprocess(self, pred_obj, pred_cls, pred_boxes, anchors):

        # (h*w*a,1)*(h*w*a,num_class)=(h*w*a,num_class)-->(h*w*a*num_class,)
        scores = th.sqrt(th.sigmoid(pred_obj)*th.sigmoid(pred_cls)).flatten()
        num_topk = min(self.topk, len(scores))

        ordered_score, ordered_ind = scores.sort(
            descending=True)  # 将置信度按从大到小排序
        topk_score = ordered_score[:num_topk]  # 返回置信度最大的前num_topk个
        topk_idx = ordered_ind[:num_topk]  # 返回前num_topk个置信度最大的索引

        mask = topk_score > self.conf  # 通过和置信度比较  消除小于阈值的
        scores = topk_score[mask]  # 前num_topk且大于阈值的 置信度 设数量m (m,)
        topk_idx = topk_idx[mask]  # 索引

        labels = topk_idx % self.num_class  # (m,)
        anchor_idx = topk_idx//self.num_class  # (m,)
        '''anchor_idx的理解 因为topk_idx是返回的是索引 从(0~h*w*a*num_clas)
        中得到 现在将索引映射到(0~h*w*a)且取整 得到的就是topk_idx在长度(h*w*a)的
         范围内新的索引 用作anchor的索引 '''

        pred_boxes = pred_boxes[anchor_idx]  # 得到符合条件的box
        anchors = anchors[anchor_idx]  # 得到符合条件的anchor

        boxes = self.decode_boxes(anchors, pred_boxes)  # (m,4)

        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()

        boxes, labels, scores = mul_cls_nms_label(boxes, labels, scores,
                                                  self.nms, self.num_class)

        return boxes, scores, labels


def build_yolo(cfg, device, conf, nms, topk, is_train):
    model = Yolo(cfg, device, conf, nms, topk, is_train)
    return model


if __name__ == '__main__':

    from config.v2 import net_param

    is_train = True
    data = th.rand(64, 3, 224, 224)
    device = 'cuda'
    data = data.to(device)
    net = build_yolo(net_param, device, 1e-5, 0.5, 10, True)
    net.to(device)
    output = net(data)
    print(output)
