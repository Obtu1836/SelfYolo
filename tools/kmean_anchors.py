from utils.dataset import VOCDection
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
'''
注释的代码 是利用循环编写如何使用kmean方法聚类anchor 优点是简单易懂 缺点是 慢
'''

@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

def collect_voc_box(path, input_size: int):
    '''从voc数据集 拉取标注框的信息并根据原图像将宽/高归一化 
    然后在放大到输入图像的尺寸

    '''
    boxes = []
    dataset = VOCDection(path, input_size, None)
    for i in range(len(dataset)):
        img = dataset.pull_image(i)
        img_h, img_w = img.shape[:2]
        _, box = dataset.pull_anno(i)

        for box_label in box:
            box = box_label[:-1]
            x1, y1, x2, y2 = box
            # bw = (x2-x1)/max(img_h, img_w)*input_size #letterbox归一化并放大到输入图像的尺寸
            # bh = (y2-y1)/max(img_h, img_w)*input_size
            bw=(x2-x1)/img_w*input_size # 这是不加入letterbox时的归一化和缩放
            bh=(y2-y1)/img_h*input_size
            if bw < 1 or bh < 1:
                continue
            boxes.append(Box(0, 0, bw, bh))
    print('拉取所有图片的标注框结束')
    return boxes

def box_to_array(boxes:list[Box])->NDArray:

    boxs=[[box.w,box.h] for box in boxes]
    boxs=np.array(boxs)
    return boxs

def update_cents(boxes:NDArray,cents:NDArray,k:int) ->tuple[NDArray,float]:
    
    N,M=boxes.shape
    bw=boxes[:,0][:,None] #(N,1)
    bh=boxes[:,1][:,None]

    cw=cents[:,0][None,:] #(1,k)
    ch=cents[:,1][None,:]

    inw=np.minimum(bw,cw) #(N,K)
    inh=np.minimum(bh,ch) #(N,K)
    in_area=inw*inh #(N,K)

    barea=bw*bh #(N,1)
    carea=cw*ch #(1,K)
    union=barea+carea-in_area #(N,K)
    iou=in_area/(union+1e-8)
    dist=1-iou #(N,K)

    min_dist_ind=np.argmin(dist,axis=1) #(N,)
    mid_dist=dist[np.arange(len(min_dist_ind)),min_dist_ind]
    loss=mid_dist.sum()

    counts=np.bincount(min_dist_ind,minlength=k)
    new_cents=np.zeros((k,M))
    np.add.at(new_cents,min_dist_ind,boxes)
    
    empty=counts==0
    not_empty=counts>0
    new_cents[not_empty]/=counts[not_empty][:,None]

    if np.any(empty):
        num=empty.sum()
        new_cents[empty]=boxes[np.random.choice(N,num)]
    
    return new_cents,loss


def anchor_box_kmean(boxes:NDArray,k:int,rtol:float,
                     max_iters:int=20000):

    N=len(boxes)
   
    cents=boxes[np.random.choice(N,k)]

    init_loss=np.inf
    i=1
    while True:
        cents,loss=update_cents(boxes,cents,k)
        i+=1
        if abs(loss-init_loss)<rtol or i>max_iters :
            break
        init_loss=loss

    return cents


# def anchor_box_kmean(boxes: list[Box], k: int, iters: int, rtol: float = 1e-6):
#     cents = []

#     length = len(boxes)
#     samples = random.sample(range(length), k)
#     for i in samples:
#         cents.append(boxes[i])

#     cents,  old_loss = execute_kmean(k, boxes, cents)
#     iter_num = 1
#     while True:
#         cents, loss = execute_kmean(k, boxes, cents)
#         iter_num += 1
#         if abs(loss-old_loss) < rtol or iter_num > iters:
#             break
#         old_loss = loss

#     for ct in cents:
#         print('w,h:', round(ct.w, 2), round(ct.h, 2),
#               'area:', round(ct.w, 2)*round(ct.h, 2))

#     return cents


# def execute_kmean(k: int, boxes: list[Box], cents: list[Box]):

#     loss = 0
#     groups = []
#     new_cents = []

#     for i in range(k):
#         groups.append([])
#         new_cents.append(Box(0, 0, 0, 0))

#     for box in boxes:
#         min_dis = float('inf')
#         group_index = 0

#         for ct_id, ct in enumerate(cents):
#             distance = (1-iou(box, ct))
#             if distance < min_dis:
#                 min_dis = distance
#                 group_index = ct_id

#         groups[group_index].append(box)
#         loss += min_dis
#         new_cents[group_index].w += box.w
#         new_cents[group_index].h += box.h

#     for i in range(k):
#         new_cents[i].w /= max(len(groups[i]), 1)
#         new_cents[i].h /= max(len(groups[i]), 1)

#     return new_cents,  loss


# def iou(box1: Box, box2: Box):
#     x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
#     x2, y2, w2, h2 = box2.x, box2.y, box2.w, box2.h
#     s1 = w1*h1
#     s2 = w2*h2

#     minx1, miny1 = x1-w1/2, y1-h1/2
#     maxx1, maxy1 = x1+w1/2, y1+h1/2
#     minx2, miny2 = x2-w2/2, y2-h2/2
#     maxx2, maxy2 = x2+w2/2, y2+h2/2

#     l, r = max(minx1, minx2), min(maxx1, maxx2)
#     w = r-l
#     u, d = max(miny1, miny2), min(maxy1, maxy2)
#     h = d-u
#     if w < 0 or h < 0:
#         return 0
#     inter = w*h
#     iou = inter/(s1+s2-inter)
#     return iou

def main():
    path = r"D:\program\VOCdevkit"
    boxes=collect_voc_box(path,416)
    boxes=box_to_array(boxes)
    anchors=anchor_box_kmean(boxes,5,1e-6)
    # anchors=anchor_box_kmean(boxes,5,20000,plus=False)
    print(anchors)

if __name__ == '__main__':
    main()
