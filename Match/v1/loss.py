from torch.nn import functional as f
from .matcher import Matcher
from tools.NMS import cal_giou

class Criterion:
    def __init__(self,
                 weight:list[float],
                 num_class:int,
                 ):
        self.weight_obj, self.weight_cls, self.weight_boxes = weight
        # self.weight_noobj = 0.5
        self.matcher = Matcher(num_class)
        self.num_class = num_class

    def __call__(self, pred, target):
        device = pred['pred_cls'][0].device
        stride = pred['stride']
        fmp_size = pred['fmp_size']

        pred_obj = pred['pred_obj'].view(-1)  # (b*m,)
        pred_cls = pred['pred_cls'].view(-1, self.num_class)  # (b*m,num_class)
        pred_boxes = pred['pred_bboxes'].view(-1, 4)  # (b*m,4)

        gt_obj, gt_cls, gt_boxes = self.matcher(fmp_size, stride, target)
        gt_obj = gt_obj.view(-1).to(device).float()
        gt_cls = gt_cls.view(-1, self.num_class).to(device).float()
        gt_boxes = gt_boxes.view(-1, 4).to(device).float()

        pos_mask = gt_obj > 0 #正样本
        # neg_mask = ~pos_mask #负样本
        num_pos = pos_mask.sum().clamp(min=1) # 正样本的个数
        # num_neg = neg_mask.sum().clamp(min=1) # 负样本的个数

        # loss_all = self._calculate_obj_loss(pred_obj, gt_obj)  # per-cell loss
        # loss_pos = loss_all[pos_mask].sum() / num_pos
        # loss_neg = loss_all[neg_mask].sum() / num_neg
        # loss_obj = loss_pos + self.weight_noobj * loss_neg
        loss_obj=self._calculate_obj_loss(pred_obj,gt_obj)
        loss_obj=loss_obj.sum()/num_pos


        pred_cls_pos = pred_cls[pos_mask]
        gt_cls_pos = gt_cls[pos_mask]
        loss_cls = self._calculate_cls_loss(pred_cls_pos, gt_cls_pos)
        loss_cls = loss_cls.sum() / num_pos

        pred_box_pos = pred_boxes[pos_mask]
        gt_boxes_pos = gt_boxes[pos_mask]
        loss_boxes = self._calculate_box_loss(pred_box_pos, gt_boxes_pos)
        loss_boxes = loss_boxes.sum() / num_pos

        total_loss = self.weight_obj * loss_obj + self.weight_cls * loss_cls + \
                     self.weight_boxes * loss_boxes

        loss_dict = {'loss_obj': loss_obj,
                     'loss_cls': loss_cls,
                     'loss_boxes': loss_boxes,
                     'total_loss': total_loss}

        return loss_dict

    def _calculate_obj_loss(self, pred_obj, gt_obj):
        loss = f.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')
        return loss

    def _calculate_cls_loss(self, pred_cls, gt_cls):
        loss = f.binary_cross_entropy_with_logits(pred_cls, gt_cls, reduction='none')
        return loss

    def _calculate_box_loss(self, pred_boxes, gt_boxes):
        giou = cal_giou(pred_boxes, gt_boxes)
        loss = 1 - giou
        return loss

def build_criterion(weights:list[float], num_class:int,):
    return Criterion(weights, num_class)






