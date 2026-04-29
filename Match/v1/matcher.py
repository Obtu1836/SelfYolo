

import torch as th
import numpy as np


class Matcher:
    '''
    将标注的信息 按照网络输出特征图的大小 放缩到特征图范围内进行匹配 以供下一步loss计算
    网络输出 以供包括三部分损失 1 当前特征图网格是否包含正样本
                            2 类别损失
                            3 框 损失
    
    网络输出的box形状为[cx,cy,w,h]

    '''
    def __init__(self, num_class: int):

        self.num_class = num_class

    @th.no_grad()
    def __call__(self,
                 fmp_size: tuple[int, int],
                 stride: int,
                 targets: dict
                 ):
        bs = len(targets)
        h, w = fmp_size

        gt_obj = np.zeros((bs, h, w, 1), dtype=np.float32)  # (b,h,w,1)
        gt_cls = np.zeros((bs, h, w, self.num_class),
                          dtype=np.float32)  # (b,h,w,num_class)
        gt_boxes = np.zeros((bs, h, w, 4), dtype=np.float32)  # (b,h,w,4)

        for batch in range(bs):#遍历batch 每个batch 包括batch张图片
            tgt_per_image = targets[batch] 
            tgt_cls = tgt_per_image['labels'].cpu().numpy()# 获取每张图片里所有物体的类别
            tgt_boxes = tgt_per_image['boxes'].cpu().numpy()#获取每张图片所有物体的标注框
            for gt_box, gt_label in zip(tgt_boxes, tgt_cls): #遍历每个标注框和类别 
                x1, y1, x2, y2 = gt_box
                bw, bh = x2-x1, y2-y1
                if bw < 1 or bh < 1:
                    continue
                xc, yc = (x1+x2)/2, (y1+y2)/2
                gridx, gridy = int(xc/stride), int(yc/stride)

                if not (0 <= gridx < w and 0 <= gridy < h):
                    continue
                gt_obj[batch, gridy, gridx] = 1.0
                '将标注的框 映射到特征图(gridy,gridx)处并标记为正样本'

                cls_one_hot = np.zeros(self.num_class, dtype=np.float32)
                cls_one_hot[gt_label] = 1.0
                gt_cls[batch, gridy, gridx] = cls_one_hot
                '将标注的物体类别映射到特征图(gridy,gridx)处 并将类别onehot 在正确类别标记为1'

                gt_boxes[batch, gridy, gridx] = np.array(
                    [x1, y1, x2, y2], dtype=np.float32)
                '记录真实的框的位置 因为网络输出时 输出的预测的真实的框的位置'
        
        gt_obj=th.from_numpy(gt_obj).reshape(bs,-1,1).float()
        gt_cls=th.from_numpy(gt_cls).reshape(bs,-1,self.num_class).float()
        gt_boxes=th.from_numpy(gt_boxes).reshape(bs,-1,4).float()

        return gt_obj,gt_cls,gt_boxes
    

if __name__ == '__main__':
    
    from  config.v1 import dataset_param
    from utils.dataset import build_dataloader,build_datasets,build_transform

    img_size=640
    transform=build_transform(img_size,True)
    dataset=build_datasets(dataset_param.basepath,transform,img_size,False)
    train_loader=build_dataloader(dataset,64,8)

    matcher=Matcher(20)

    for image,target in train_loader:
        gt_obj,gt_cls,gt_boxes=matcher((img_size//32,img_size//32),32,target)
        print(gt_cls[0][1])
        break

