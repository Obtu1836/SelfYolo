import torch as th
import torch.nn.functional as f
from tools.NMS import cal_giou
from .matcher import Matcher

class Crieterion:
    def __init__(self,weights:list[float],
                    num_class:int,
                    num_anchors:int,
                    anchor_size:list[list[float]],
                    anchor_base_size:int,
                    iou:float=0.5):

        self.obj_weight,self.cls_weight,self.boxes_weight=weights
        self.num_class=num_class
        
        self.matcher=Matcher(num_class,num_anchors,anchor_size,anchor_base_size,iou)
    
    def __call__(self,preds:dict,target:dict,epoch=0):
        device=preds['pred_cls'][0].device
        fpn_stride=preds['stride']
        fmp_size = preds['fmp_size']

        pred_obj=th.cat(preds['pred_obj'],dim=1).view(-1)
        pred_cls=th.cat(preds['pred_cls'],dim=1).view(-1,self.num_class)
        pred_boxes = th.cat(preds['pred_boxes'],dim=1).view(-1,4)

        gt_obj,gt_cls,gt_boxes=self.matcher(fmp_size,fpn_stride,target)
        gt_obj=gt_obj.view(-1).to(device)
        gt_cls=gt_cls.view(-1,self.num_class).to(device)
        gt_boxes=gt_boxes.view(-1,4).to(device)

        pos_mask=gt_obj>0
        pos_num=pos_mask.sum().clamp(min=1)

        obj_loss=self.cal_obj_loss(pred_obj,gt_obj)
        obj_loss=obj_loss.sum()/pos_num

        pred_boxes_pos=pred_boxes[pos_mask]
        gt_boxes_pos=gt_boxes[pos_mask]
        boxes_loss,ious=self.cal_boxes_loss(pred_boxes_pos,gt_boxes_pos)
        boxes_loss=boxes_loss.sum()/pos_num

        pred_cls_pos=pred_cls[pos_mask]
        gt_cls_pos=gt_cls[pos_mask]*ious.unsqueeze(-1).clamp(0)
        cls_loss=self.cal_cls_loss(pred_cls_pos,gt_cls_pos)
        cls_loss=cls_loss.sum()/pos_num


        loss=self.obj_weight*obj_loss+\
               self.cls_weight*cls_loss+\
               self.boxes_weight*boxes_loss
        
        total_loss = {'loss_obj': obj_loss,
                      'loss_cls': cls_loss,
                      'loss_boxes': boxes_loss,
                      'total_loss': loss}

        return total_loss
    
    def cal_obj_loss(self,pred,target):
        loss=f.binary_cross_entropy_with_logits(pred,target,reduction='none')
        return loss
    def cal_cls_loss(self,pred,target):
        loss=f.binary_cross_entropy_with_logits(pred,target,reduction='none')
        return loss
    
    def cal_boxes_loss(self,pred,target):

        iou=cal_giou(pred,target)
        return 1-iou,iou


def build_criterion(weights, num_class, num_anchors, anchor_size, anchor_base_size):
    criterion = Crieterion(weights, num_class, num_anchors, anchor_size, anchor_base_size)
    return criterion







