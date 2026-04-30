import torch as th
import numpy as np
from numpy.typing import NDArray


class Matcher:
    def __init__(self, num_class: int,
                 iou_thresh: float,
                 anchor_size: list):

        self.num_class = num_class
        self.iou_thresh = iou_thresh
        self.anchor_size = anchor_size
        self.num_anchor = len(anchor_size)

        self.anchor_box = np.array([[0, 0, anchor[0], anchor[1]]
                                   for anchor in anchor_size], dtype=np.float32)

    @th.no_grad()
    def __call__(self, fmp_size: th.Tensor, stride: int, targets: list[dict[str, NDArray]]):
        '''
        按batch 拼接的标签组 在dataset.py中 一个batch的标签以列表内元素为字典的格式返回
        【{'boxes':[[N,4],[N,4]] 二维数组,
          {"labels':[1,2] }
        '''

        bs = len(targets)
        fmp_h, fmp_w = fmp_size

        gt_objness = np.zeros(
            (bs, fmp_h, fmp_w, self.num_anchor, 1), dtype=np.float32)
        gt_clsness = np.zeros(
            (bs, fmp_h, fmp_w, self.num_anchor, self.num_class), dtype=np.float32)
        gt_regness = np.zeros(
            (bs, fmp_h, fmp_w, self.num_anchor, 4), dtype=np.float32)

        for bat in range(bs):
            per_image_target = targets[bat]  # 获取batch内每一张图片的标注信息
            tgt_cls = per_image_target['labels']
            tgt_boxes = per_image_target['boxes']

            for gt_box, gt_label in zip(tgt_boxes, tgt_cls):
                x1, y1, x2, y2 = gt_box  # 获取标注框的坐标
                xc, yc = (x1+x2)/2, (y1+y2)/2  # 获取中心点
                bw, bh = x2-x1, y2-y1  # 计算宽高
                gt_box = [0, 0, bw, bh]  # 转化为中心点重合形式用于计算和anchor的iou
                if bw < 1 or bh < 1:
                    continue
                iou = self.compute_iou(self.anchor_box, gt_box)  # (N,)
                iou_mask = iou > self.iou_thresh  # 通过bool值筛选iou大于阈值的
                label_results = []  # 保存符合的坐标 (gy,gx,anchor)
                if iou_mask.sum() == 0:
                    # 如果没有符合条件的 就挑选一个iou最大的作为正样本
                    iou_ind = np.argmax(iou)
                    gridx = int(xc/stride)
                    gridy = int(yc/stride)
                    label_results.append([gridx, gridy, iou_ind])
                else:
                    # 如果符合条件的>=1,那么都选上
                    for iou_id, iou_m in enumerate(iou_mask):
                        if iou_m:
                            gridx = int(xc/stride)
                            gridy = int(yc/stride)
                            label_results.append([gridx, gridy, iou_id])

                for result in label_results:  # 遍历所有符合条件的 并将这些标记为正样本
                    gx, gy, idx = result
                    if 0 <= gx < fmp_w and 0 <= gy < fmp_h:
                        gt_objness[bat, gy, gx, idx] = 1.0

                        cls_one_hot = np.zeros(self.num_class)
                        cls_one_hot[int(gt_label)] = 1.0
                        gt_clsness[bat, gy, gx, idx] = cls_one_hot

                        gt_regness[bat, gy, gx, idx] = np.array(
                            [x1, y1, x2, y2])
                    '''相比于yolov1 此处多了anchor 也对应网络的输出多了anchor'''

        # gt_objness=gt_objness.reshape(bs,-1,1) #(b,h*w*k,1)
        # gt_clsness=gt_clsness.reshape(bs,-1,self.num_class)
        # gt_regness=gt_regness.reshape(bs,-1,4)

        gt_objness = th.from_numpy(gt_objness)
        gt_clsness = th.from_numpy(gt_clsness)
        gt_boxness = th.from_numpy(gt_regness)

        return gt_objness, gt_clsness, gt_boxness

    def compute_iou(self, anchor_box, gt_box):
        '''
        计算1个标注框 与聚类anchor的iou anchor.shape=(N,4)
                                     gt_box.shape=(4,)
        先将所有框 变形为 中心点重合的形式 然后计算iou
        '''
        # 先将anchor[0,0,w,h]转成(x1,y1,x2,y2)格式
        new_anchor_box = np.zeros_like(anchor_box)
        new_anchor_box[:, :2] = new_anchor_box[:, :2]-anchor_box[:, 2:]/2
        new_anchor_box[:, 2:] = new_anchor_box[:, 2:]+anchor_box[:, 2:]/2
        new_anchor_area = anchor_box[:, 2]*anchor_box[:, 3]  # (N,)

        # 将gt_box转化为(0,0,bw,bh)->（x1,y1,x2,y2)
        gt_box = np.array(gt_box)  # [0,0,bw,bh]
        gt_box[:2] = gt_box[:2]-gt_box[2:]/2  # (-bw/2,-bh/2,bw,bh)
        gt_box[2:] = gt_box[:2]+gt_box[2:]  # (-bw/2,-bh/2,bw/2,bh/2)
        gt_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])

        '''
        以上两步 将anchor中的每一个框和当前标注框都转化为中心点重合的形式
        并且通过宽和高 计算出(x1,y1,x2,y2)的形式
        '''

        l = np.maximum(gt_box[0], new_anchor_box[:, 0])
        r = np.minimum(gt_box[2], new_anchor_box[:, 2])
        w = np.maximum(r-l, 0)

        u = np.maximum(gt_box[1], new_anchor_box[:, 1])
        d = np.minimum(gt_box[3], new_anchor_box[:, 3])
        h = np.maximum(d-u, 0)

        inter_area = w*h

        union = new_anchor_area+gt_area-inter_area
        iou = inter_area/(union+1e-8)
        iou = np.clip(iou, a_min=0, a_max=1)

        return iou


if __name__ == '__main__':

    # anchor = [[38, 64],
    #           [89, 147],
    #           [145, 285],
    #           [258, 169],
    #           [330, 340]]

    # matcher=Matcher(20,0.1,anchor)

    # target=[{'boxes':np.random.randint(0,100,size=(10,4)),
    #          'labels':np.random.randint(0,20,size=(10,))}]
    from config.v1 import dataset_param
    from config.v2 import net_param
    from utils.dataset import build_dataloader, build_datasets, build_transform

    img_size = 640
    transform = build_transform(img_size, True)
    dataset = build_datasets(dataset_param.basepath,
                             transform, img_size, False)
    train_loader = build_dataloader(dataset, 64, 8)

    matcher = Matcher(20, 0.5, net_param.anchor_size)
    fmp_size = th.Tensor([20, 20])
    for image, target in train_loader:
        gt_obj, gt_cls, gt_boxes = matcher(fmp_size, 32, target)

        print(gt_cls[0][1])
        break

    # a,b,c=matcher(13,32,target)
    # print(a.dtype)
