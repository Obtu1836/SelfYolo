import torch as th
import numpy as np
from numpy.typing import NDArray


class Matcher:
    def __init__(self, num_class: int, num_anchors: int,
                 anchor_size, anchor_base_size: int,
                 iou_thresh: float = 0.5):

        self.num_class = num_class
        self.num_anchors = num_anchors
        self.iou_thresh = iou_thresh
        self.anchor_base_size = float(anchor_base_size)
        self.anchor_wh = np.asarray(
            anchor_size, dtype=np.float32).reshape(-1, num_anchors, 2)

    def get_level_anchor_boxes(self, level: int, fmp_size, stride: int):
        fmp_h, fmp_w = fmp_size
        current_h = float(fmp_h * stride)
        current_w = float(fmp_w * stride)

        anchor_wh = self.anchor_wh[level].copy()
        anchor_wh[:, 0] *= current_w / self.anchor_base_size
        anchor_wh[:, 1] *= current_h / self.anchor_base_size

        zeros = np.zeros((self.num_anchors, 2), dtype=np.float32)
        return np.concatenate([zeros, anchor_wh], axis=1)

    @th.inference_mode()
    def __call__(self, fmp_size, fpn_stride, targets):
        '''
        fmp_size=[[52,52],[26,26],[13,13]]
        fpn_stride=[8,16,32]

        标签匹配策略：
        遍历每个batch中的每张图片 对每一个图片中的标注框与所有的anchor计算交并比(方式是中心点重合)选择交并比最大的anchor
        作为预测该物体的框 在根据框的索引确定出是哪个层级(level)的特征图索引。在这一过程中 就实现了 大物体使用大的anchor
        小物体使用小的anchor 在根据标注信息以及level 确定出x1,y1,x2,y2

        在确定出层级时 如果某个标注框与所有的anchor匹配达不到阈值 就挑选iou最大的作为目标
        如果有多个anchor与标注框的iou达到阈值 那么都看作是正样本

        '''

        level_anchor_boxes = [
            self.get_level_anchor_boxes(
                level, fmp_size[level], fpn_stride[level])
            for level in range(len(fmp_size))
        ]

        all_anchor_boxes = np.concatenate(level_anchor_boxes, axis=0)
        assert len(fmp_size) == len(fpn_stride)
        bs = len(targets)

        gt_objs = [th.zeros(bs, fh, fw, self.num_anchors, 1)
                   for (fh, fw) in fmp_size]  # 根据特征图大小分别建立
        gt_clss = [th.zeros(bs, fh, fw, self.num_anchors, self.num_class)
                   for (fh, fw) in fmp_size]
        gt_boxes = [th.zeros(bs, fh, fw, self.num_anchors, 4)
                    for (fh, fw) in fmp_size]

        for batch in range(bs):  # 遍历batch
            target_pei_image = targets[batch]  # 获取图片
            tgt_cls = target_pei_image['labels'].numpy()
            tgt_boxs = target_pei_image['boxes'].numpy()

            for box, label in zip(tgt_boxs, tgt_cls,):  # 遍历单张图片中的所有标注框
                x1, y1, x2, y2 = box.tolist()
                xc, yc = (x1+x2)/2, (y1+y2)/2
                bw, bh = x2-x1, y2-y1
                g_box = [0, 0, bw, bh]  # 先转化为左上角对齐
                if bw < 1 or bh < 1:
                    continue
                # 内部会将左上角对齐转化为中心点对齐 计算iou
                iou = self.compute_iou(all_anchor_boxes, g_box)
                # 过滤 挑选出iou>阈值的 （单个标注框与所有anchor)
                iou_mask = iou >= self.iou_thresh

                label_results = []
                if iou_mask.sum() == 0:  # 如果不存在iou>阈值的 就挑选iou最大的
                    iou_idx = np.argmax(iou)
                    level = iou_idx//self.num_anchors  # 确定层级
                    # 确定层级下是哪个anchor 因为共分为3个层级 每个层级又分为3个anchor
                    anchor_dx = iou_idx-level*self.num_anchors
                    stride = fpn_stride[level]  # 确定下采样步长

                    xc_s = xc/stride  # 根据步长将标注的中心点映射到特征图
                    yc_s = yc/stride

                    gridx = int(xc_s)
                    gridy = int(yc_s)

                    label_results.append(
                        [gridx, gridy, level, anchor_dx])  # 确定那个坐标点位
                else:
                    # 如果多个iou>阈值，通过遍历将他们都看作是正样本
                    for iou_id, iou_m in enumerate(iou_mask):
                        if iou_m:
                            level = iou_id//self.num_anchors
                            anchor_dx = iou_id-level*self.num_anchors
                            stride = fpn_stride[level]

                            xc_s = xc/stride
                            yc_s = yc/stride
                            gridx = int(xc_s)
                            gridy = int(yc_s)

                            label_results.append(
                                [gridx, gridy, level, anchor_dx])  # 确定正样本点位

                for result in label_results:  # 制作正样本标签
                    gx, gy, lv, ad = result
                    fh, fw = fmp_size[lv]
                    if gx < fw and gy < fh:
                        gt_objs[lv][batch, gy, gx, ad] = 1.0
                        cls_one_hot = th.zeros(self.num_class)
                        cls_one_hot[int(label)] = 1.0
                        gt_clss[lv][batch, gy, gx, ad] = cls_one_hot
                        gt_boxes[lv][batch, gy, gx, ad] = th.as_tensor(
                            [x1, y1, x2, y2])

        gt_objs = th.cat([gt.view(bs, -1, 1) for gt in gt_objs], dim=1).float()
        gt_clss = th.cat([gt.view(bs, -1, self.num_class)
                         for gt in gt_clss], dim=1).float()
        gt_boxes = th.cat([gt.view(bs, -1, 4)
                          for gt in gt_boxes], dim=1).float()

        return gt_objs, gt_clss, gt_boxes

    def compute_iou(self, anchor: NDArray, box):
        '''
        将anchor和box转化为中心点重合的方式 进行交并比计算
        '''
        anchor = np.asarray(anchor, dtype=np.float32)
        box = np.asarray(box, dtype=np.float32)

        anchor_wh = anchor[:, 2:4]
        anchor_xyxy = np.concatenate(
            [-anchor_wh / 2.0, anchor_wh / 2.0], axis=1
        )
        anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]

        gt_wh = np.repeat(box[None, 2:4], anchor.shape[0], axis=0)
        gt_xyxy = np.concatenate(
            [-gt_wh / 2.0, gt_wh / 2.0], axis=1
        )
        gt_area = gt_wh[:, 0] * gt_wh[:, 1]

        lt = np.maximum(anchor_xyxy[:, :2], gt_xyxy[:, :2])
        rb = np.minimum(anchor_xyxy[:, 2:], gt_xyxy[:, 2:])
        inter_wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        union = anchor_area + gt_area - inter_area
        iou = inter_area / (union + 1e-8)
        return np.clip(iou, a_min=1e-10, a_max=1.0)


if __name__ == '__main__':
    from config.v1 import dataset_param
    from config.v3 import cfg
    from utils.dataset import build_dataloader, build_datasets, build_transform

    img_size = 640
    transform = build_transform(img_size, True)
    dataset = build_datasets(dataset_param.basepath,
                             transform, img_size, False)
    train_loader = build_dataloader(dataset, 64, 8)

    anchor = cfg['net']['anchor_size']
    matcher = Matcher(20, 3, anchor,416)
    fmp_size = [[52, 52], [26, 26], [13, 13]]
    for image, target in train_loader:
        gt_obj, gt_cls, gt_boxes = matcher(fmp_size, [8, 16, 32], target)

        print(gt_cls[0][1])
        break
