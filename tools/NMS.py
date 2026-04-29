import numpy as np
from numpy.typing import NDArray
from typing import Literal
import torch as th


def nms(bboxes: NDArray, scores: NDArray, nms_thresh: float):

    x1, y1, x2, y2 = np.split(bboxes, indices_or_sections=[1, 2, 3], axis=1)
    areas = (x2-x1)*(y2-y1)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        high, other = order[0], order[1:]
        keep.append(high)
        xx1 = np.maximum(x1[high], x1[other])
        yy1 = np.maximum(y1[high], y1[other])
        xx2 = np.minimum(x2[high], x2[other])
        yy2 = np.minimum(y2[high], y2[other])

        w = np.maximum(1e-8, xx2-xx1)
        h = np.maximum(1e-8, yy2-yy1)

        inter = w*h
        union = areas[high]+areas[other]-inter
        iou = inter/union

        idx = np.where(iou <= nms_thresh)[0]
        order = order[idx+1]

    return np.array(keep)


def multi_cls_nms(bboxes: NDArray, scores: NDArray, conf_thresh: float,
                  nms_thresh: float, num_class: int):

    labels = np.argmax(scores, axis=1)
    confidence = scores[np.arange(len(scores)), labels]

    idx = np.where(confidence >= conf_thresh)[0]
    bboxes = bboxes[idx]
    confidence = confidence[idx]
    labels = labels[idx]

    keep = np.zeros(len(labels), dtype=bool)
    for i in range(num_class):
        ind = np.where(labels == i)[0]
        if len(ind) == 0:
            continue
        cls_bboxes = bboxes[ind]
        cls_conf = confidence[ind]
        idx = nms(cls_bboxes, cls_conf, nms_thresh)
        keep[ind[idx]] = True

    bbox = bboxes[keep]
    score = confidence[keep]
    label = labels[keep]

    return bbox, score, label


def cal_giou(boxes1: th.Tensor, boxes2: th.Tensor, mode: Literal['giou', 'iou'] = 'giou'):
    '''
    计算两个长度相等 相同索引的框的loss
    iou和giou都能反应框的重叠程度,但是giou会额外考虑两个框空出来的面积 也就是最小外接矩形
    比iou更严格 而且 当两个框不重叠时 iou=0,不能反应两个框是稍微远 还是很远 giou可能会变成负数
    并且两个框越远 外接框面积越大 giou越小

    公式 iou-(c-u)/c   c:外接面积 u 两个框的并集
    '''
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    eps = th.finfo(boxes1.dtype).eps
    box1_wh = (boxes1[..., 2:]-boxes1[..., :2]).clamp(min=0)
    box2_wh = (boxes2[..., 2:]-boxes2[..., :2]).clamp(min=0)

    box1_area = box1_wh[..., 0]*box1_wh[..., 1]
    box2_area = box2_wh[..., 0]*box2_wh[..., 1]

    inter_x1 = th.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = th.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = th.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = th.min(boxes1[..., 3], boxes2[..., 3])

    inter_wh = th.stack((inter_x2-inter_x1, inter_y2 -
                        inter_y1), dim=-1).clamp(min=0)
    inter_area = inter_wh[..., 0]*inter_wh[..., 1]

    uniou = box1_area+box2_area-inter_area
    iou = inter_area/uniou.clamp(min=eps)
    if mode == 'iou':
        return iou

    external_x1 = th.min(boxes1[..., 0], boxes2[..., 0])  # 最小外接矩形确定形状
    external_y1 = th.min(boxes1[..., 1], boxes2[..., 1])
    external_x2 = th.max(boxes1[..., 2], boxes2[..., 2])
    external_y2 = th.max(boxes1[..., 3], boxes2[..., 3])

    external_wh = th.stack(
        (external_x2-external_x1, external_y2-external_y1), dim=-1).clamp(min=0)
    external_area = (external_wh[..., 0]*external_wh[..., 1]).clamp(min=eps)
    giou = iou-(external_area-uniou)/external_area
    giou = th.clamp(giou, min=-1.0, max=1.0)
    
    # giou = th.nan_to_num(giou, nan=-1.0)   # 或 nan=0.0，按你偏好选择

    return giou


if __name__ == '__main__':

    box1 = th.tensor([[0, 0, 5, 5],
                      [10, 15, 30, 45]])

    box2 = th.tensor([[1, 1, 6, 6],
                      [15, 20, 35, 50]])

    res = cal_giou(box1, box2, 'iou')
    print(res)
