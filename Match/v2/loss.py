import torch as th
from torch.nn import functional as f
from .matcher import Matcher
from config.v2 import NetParam, net_param
from tools.NMS import cal_giou


class Critierion:
    def __init__(self, net_param: NetParam, iou=0.5):

        self.num_class = net_param.num_class
        self.obj_weight, self.cls_weight, self.boxes_weight = net_param.weights

        self.matcher = Matcher(self.num_class, 0.5, net_param.anchor_size)

    def cal_loss_obj(self, pred_obj, gt_obj):
        loss = f.binary_cross_entropy_with_logits(
            pred_obj, gt_obj, reduction='none')
        return loss

    def cal_loss_boxes(self, pred_boxes, gt_boxes):
        ious = cal_giou(pred_boxes, gt_boxes)
        loss = 1-ious
        return loss, ious

    def cal_loss_cls(self, pred_cls, gt_cls):
        loss = f.binary_cross_entropy_with_logits(
            pred_cls, gt_cls, reduction='none')
        return loss

    def __call__(self, outputs, targets, epoch=None):

        device = outputs['pred_obj'].device
        stride = outputs['stride']
        fmp_size = outputs['fmp_size']

        pred_obj = outputs['pred_obj'].view(-1)
        pred_cls = outputs['pred_cls'].view(-1, self.num_class)
        pred_boxes = outputs['pred_boxes'].view(-1, 4)

        gt_obj, gt_cls, gt_boxes = self.matcher(fmp_size, stride, targets)
        gt_obj = gt_obj.view(-1).to(device).float()
        gt_cls = gt_cls.view(-1, self.num_class).to(device).float()
        gt_boxes = gt_boxes.view(-1, 4).to(device).float()

        pos_mask = gt_obj > 0
        nums = pos_mask.sum().clamp(min=1)

        loss_obj=self.cal_loss_obj(pred_obj,gt_obj)
        loss_obj=loss_obj.sum()/nums
        # loss_obj = self.cal_loss_obj(pred_obj, gt_obj).view(-1)   # (N,)
        # pos_mask = gt_obj > 0
        # num_pos = pos_mask.sum().clamp(min=1)
        # num_neg = (~pos_mask).sum().clamp(min=1)

        # loss_pos = loss_obj[pos_mask].sum() / num_pos
        # loss_neg = loss_obj[~pos_mask].sum() / num_neg

        # noobj_weight = 0.2   # 建议从 0.1 开始调试（YOLO 常用较小的 noobj 权重）
        # loss_obj = loss_pos + noobj_weight * loss_neg

        pred_pos_boxes = pred_boxes[pos_mask]
        gt_pos_boxes = gt_boxes[pos_mask]
        loss_boxes, ious = self.cal_loss_boxes(pred_pos_boxes, gt_pos_boxes)
        loss_boxes = loss_boxes.sum()/nums

        pred_pos_cls = pred_cls[pos_mask]
        gt_pos_cls = gt_cls[pos_mask]*ious[:, None].clamp(min=0)
        loss_cls = self.cal_loss_cls(pred_pos_cls, gt_pos_cls)
        loss_cls = loss_cls.sum()/nums

        
        loss = self.obj_weight*loss_obj +\
            self.cls_weight*loss_cls +\
            self.boxes_weight*loss_boxes

        total_loss = {'loss_obj': loss_obj,
                      'loss_cls': loss_cls,
                      'loss_boxes': loss_boxes,
                      'total_loss': loss}

        return total_loss


def build_criterion(net_param: NetParam, args):
    critierion = Critierion(net_param)
    return critierion


if __name__ == '__main__':
    pass
