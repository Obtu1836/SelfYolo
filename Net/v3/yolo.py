import torch as th
from torch import nn

from .backbone import Darknet53
from .fpn import FPN
from .neck import SPPF
from .head import Head

from typing import cast

from config.v3 import net_param, NetParam
from tools.get_device import get_device
from tools.NMS import mul_cls_nms_label
from ..absnet import YOLO


class Yolo(YOLO):
    def __init__(self, cfg: NetParam,
                 shape: int,
                 device: str,
                 conf_thresh: float,
                 nms_thresh: float,
                 topk: int,
                 is_train: bool):
        super().__init__()

        self.cfg = cfg
        self.topk = topk
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.stride = [8, 16, 32]
        self.shape = shape
        self.num_anchors = 3
        self.num_class = net_param.num_class
        self.device = device

        self.is_train = is_train

        self.num_level = 3
        # self.num_anchors = len(cfg.anchor_size)//self.num_level  # 3
        self.anchor_size = th.as_tensor(cfg.anchor_size).float().\
            view(self.num_level, self.num_anchors, -1)
        self.anchor_base_size = float(cfg.anchor_base_size)

        self.backbone = Darknet53()
        out_dims = self.backbone.out_dims
        self.neck = SPPF(out_dims[-1], out_dims[-1],)
        self.fpn = FPN(out_dim=256, width=cfg.width)

        # fpn输出shape  [b, 256, 64, 64])
        #              [b, 256, 32, 32]
        #              [b, 256, 16, 16]

        self.head_dims = cast(list[int], self.fpn.out_dims)

        self.heads = nn.ModuleList([Head(cfg, head_dim, head_dim) for
                                    head_dim in self.head_dims])

        self.pred_obj_layer = nn.ModuleList([nn.Conv2d(cast(int, head.reg_out_dim), 1*self.num_anchors, 1, 1, 0, bias=True)
                                             for head in self.heads])

        self.pred_cls_layer = nn.ModuleList([nn.Conv2d(cast(int, head.cls_out_dim), self.num_class*self.num_anchors, 1, 1, 0, bias=True)
                                             for head in self.heads])

        self.pred_bboxes_layer = nn.ModuleList([nn.Conv2d(cast(int, head.reg_out_dim), 4*self.num_anchors, 1, 1, 0, bias=True)
                                                for head in self.heads])

    def forward(self, x: th.Tensor):
        if not self.is_train:
            return self.inference(x)
        else:
            bs = x.shape[0]

            pyramid_feats = self.backbone(x)  # 输出通道 [256 ,512,1024]
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])
            pyramid_feats = self.fpn(pyramid_feats)  # 输出通道统一到[256,256,256]

            all_fmp_size = []
            all_pred_obj = []
            all_pred_cls = []
            all_pred_bboxes = []

            for level, (feat, head_layer) in enumerate(zip(pyramid_feats, self.heads)):
                cls_feat, reg_feat = head_layer(feat)

                pred_obj = self.pred_obj_layer[level](
                    reg_feat)  # (b,n,h,w)  n=3 为每个尺度下的3个框
                pred_cls = self.pred_cls_layer[level](
                    cls_feat)  # (b,n*num_class,h,w)
                pred_bboxes = self.pred_bboxes_layer[level](
                    reg_feat)  # (b,4*n,h,w)
                fmp_size = pred_obj.shape[-2:]

                anchors = self.make_grid(level, fmp_size)  # (m,4) m=n*h*w

                pred_obj = th.permute(pred_obj, (0, 2, 3, 1)).contiguous().view(
                    bs, -1, 1)  # （b,n,h,w)-->(b,h,w,n)->(b,m,1)
                pred_cls = th.permute(pred_cls, (0, 2, 3, 1)).contiguous().view(
                    # (b,nc*n,h,w)->(b,h,w,nc*n)->(b,m,nc)
                    bs, -1, self.num_class)
                pred_bboxes = th.permute(pred_bboxes, (0, 2, 3, 1)).contiguous().view(
                    bs, -1, 4)  # (b,4*n,h,w)->(b,h,w,4*n)->(b,m,4)

                # (b,m,2)+(m,2)->(b,m,2)
                ct_pred = (th.sigmoid(
                    pred_bboxes[..., :2])+anchors[:, :2])*self.stride[level]
                # (b,m,2)*(m,2)->(b,m,2)
                wh_pred = th.exp(pred_bboxes[..., 2:])*anchors[:, 2:]
                x1y1 = ct_pred-wh_pred/2
                x2y2 = ct_pred+wh_pred/2
                bboxes = th.cat([x1y1, x2y2], dim=-1)  # (b,m,4)

                all_fmp_size.append(fmp_size)
                all_pred_obj.append(pred_obj)
                all_pred_cls.append(pred_cls)
                all_pred_bboxes.append(bboxes)

            outputs = {'pred_obj': all_pred_obj,
                       'pred_cls': all_pred_cls,
                       'pred_boxes': all_pred_bboxes,
                       'fmp_size': all_fmp_size,
                       'stride': self.stride}

            return outputs

    @th.inference_mode()
    def inference(self, x: th.Tensor):

        backbone = self.backbone(x)
        backbone[-1] = self.neck(backbone[-1])
        pyramid_feats = self.fpn(backbone)

        all_anchors = []
        all_pred_obj = []
        all_pred_cls = []
        all_pred_bboxes = []

        for level, (feat, head_layer) in enumerate(zip(pyramid_feats, self.heads)):
            cls_feat, reg_feat = head_layer(feat)

            pred_obj = self.pred_obj_layer[level](reg_feat)[0]  # (1,n,h,w)
            pred_cls = self.pred_cls_layer[level](
                cls_feat)[0]  # (1,n*num_class,h,w)
            pred_bboxes = self.pred_bboxes_layer[level](
                reg_feat)[0]  # (1,4*n,h,w)
            fmp_size = pred_obj.shape[-2:]

            anchors = self.make_grid(level, fmp_size)  # (m,4) m=n*h*w

            pred_obj = th.permute(pred_obj, (1, 2, 0)).contiguous().view(
                -1, 1)  # （n,h,w)-->(h,w,n)->(m,1)
            pred_cls = th.permute(pred_cls, (1, 2, 0)).contiguous().view(
                -1, self.num_class)  # (nc*n,h,w)->(h,w,nc*n)->(m,nc)
            pred_bboxes = th.permute(pred_bboxes, (1, 2, 0)).contiguous().view(
                -1, 4)  # (4*n,h,w)->(h,w,4*n)->(m,4)

            ct_pred = (th.sigmoid(
                pred_bboxes[:, :2])+anchors[:, :2])*self.stride[level]
            # (m,2)*(m,2)->(m,2)
            wh_pred = th.exp(pred_bboxes[:, 2:])*anchors[:, 2:]
            x1y1 = ct_pred-wh_pred/2
            x2y2 = ct_pred+wh_pred/2
            bboxes = th.cat([x1y1, x2y2], dim=-1)  # (m,4)

            all_pred_obj.append(pred_obj)  # 【[m,1],[m,1],[m,1]】
            all_pred_cls.append(pred_cls)  # 【[m,nc],[m,nc],[m,nc]】
            all_pred_bboxes.append(bboxes)  # 【[m,4],[m,4],[m,4]】
            all_anchors.append(anchors)

        scores, labels, boxes = self.post_process(
            all_pred_obj, all_pred_cls, all_pred_bboxes)
        return boxes, scores, labels,

    def make_grid(self, level: int, fmp_size):

        fmp_h, fmp_w = fmp_size
        anchor_size = self.anchor_size[level]  # (num_anchors, 2)

        y, x = th.meshgrid([th.arange(fmp_h), th.arange(fmp_w)], indexing='ij')
        xy = th.stack([x, y], dim=-1).float().view(-1, 2)  # (s,2)
        anchorxy = xy[:, None, :].repeat(
            1, self.num_anchors, 1).view(-1, 2).to(self.device)

        anchorwh = anchor_size[None, :, :].repeat(
            fmp_h * fmp_w, 1, 1).view(-1, 2).to(self.device)

        current_h = float(fmp_h * self.stride[level])
        current_w = float(fmp_w * self.stride[level])

        scale_h = current_h / self.anchor_base_size
        scale_w = current_w / self.anchor_base_size

        anchorwh = anchorwh.clone()
        anchorwh[:, 0] *= scale_w
        anchorwh[:, 1] *= scale_h

        anchors = th.cat([anchorxy, anchorwh], dim=-1)  # (m,4)
        return anchors

    def post_process(self, all_pred_obj, all_pred_cls, all_pred_bboxes):

        all_scores = []
        all_labels = []
        all_boxes = []

        for p_obj, p_cls, p_boxes in zip(all_pred_obj, all_pred_cls, all_pred_bboxes):
            # (m,1)*(m,nc) -->(m,nc)-->(m*nc,)
            p_score = th.sqrt(th.sigmoid(p_obj)*th.sigmoid(p_cls)).flatten()
            num_topk = min(self.topk, p_boxes.size(0))
            prob, idx = p_score.sort(descending=True)
            topk_score = prob[:num_topk]
            topk_idx = idx[:num_topk]

            keep_idx = topk_score >= self.conf_thresh
            scores = topk_score[keep_idx]
            topk_idx = topk_idx[keep_idx]

            anchor_idx = topk_idx//self.num_class
            labels = topk_idx % self.num_class

            bboxes = p_boxes[anchor_idx]
            all_scores.append(scores)
            all_labels.append(labels)
            all_boxes.append(bboxes)

        scores = th.cat(all_scores)
        labels = th.cat(all_labels)
        boxes = th.cat(all_boxes)

        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        boxes = boxes.cpu().numpy()

        boxes, labels, scores = mul_cls_nms_label(
            boxes, labels, scores, self.nms_thresh, self.num_class)

        return scores, labels, boxes


def build_yolo(cfg, shape, device, conf_thresh, nms_thresh, topk, is_train):
    net = Yolo(cfg, shape, device, conf_thresh, nms_thresh, topk, is_train)
    return net


if __name__ == '__main__':

    device = get_device('cuda')
    shape = 416
    net = Yolo(net_param, shape, device, 0.05, 0.2, 100, False)
    data = th.rand(1, 3, shape, shape)

    data = data.to(device)
    net.to(device)

    a, b, c = net(data)
    print(a.shape)
    print(b.shape)
    print(c.shape)
