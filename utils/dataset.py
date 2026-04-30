import cv2
import numpy as np
import torch as th
from xml.etree import ElementTree as ET

from typing import Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from .transforms import build_transform

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class Anntation:
    '''
    获取单张图片的的标注信息 包括 框和类别的信息
    '''

    def __init__(self, difficult=False):
        self.class_idx = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.difficult = difficult

    def __call__(self, target):

        coord_class_idx = []
        for obj in target.iter('object'):  # 遍历 字段object
            diffi = int(obj.find('difficult').text) == 1  # 排除标签 difficult的
            if not self.difficult and diffi:
                continue
            name = obj.find('name').text.lower().strip()  # 获取类别名称
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for pt in pts:  # 遍历4个点坐标
                cur_pt = int(bbox.find(pt).text)-1  # voc标注位置时 是从像素1开始的
                bndbox.append(cur_pt)
            label_idx = self.class_idx.get(name)  # 将文字形式转化为数字
            bndbox.append(label_idx)
            coord_class_idx.append(bndbox)

        return coord_class_idx  # [[x1,y1,x2,y2,0],...]


class VOCDection(Dataset):
    def __init__(self, basepath, img_size: int, transform: Callable | None,
                 img_sets=[('2007', 'trainval'), ('2012', 'trainval')]):

        self.basepath = Path(basepath)
        self.img_size = img_size
        self.target_transform = Anntation()

        self.anno_path = "{path}/Annotations/{name}.xml"
        self.image_path = "{path}/JPEGImages/{name}.jpg"

        if transform is not None:
            self.transform = transform

        self.ids = list()

        for year, name in img_sets:
            rootpath = Path.joinpath(self.basepath, 'VOC'+year)
            for line in open(Path.joinpath(rootpath, 'ImageSets', 'Main', name+'.txt')):
                self.ids.append((rootpath, line.strip()))

        # self.ids=[('/Users/mac/program/VOCdevkit/VOC2012','000001'),...]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target

    def pull_item(self, index):
        image, target = self.load_image_target(index)
        image, target = self.transform(image, target)
        return image, target

    def pull_image(self, index):
        path, name = self.ids[index]
        img = cv2.imread(self.image_path.format(path=path, name=name))
        assert img is not None
        return img

    def pull_anno(self, index):
        path, name = self.ids[index]
        ann = ET.parse(self.anno_path.format(path=path, name=name)).getroot()
        gt = self.target_transform(ann)

        return name, gt

    def load_image_target(self, index):
        '''返回每一张照片和这张照片中所有的类别和标注框'''
        path, name = self.ids[index]
        image_path = Path(self.image_path.format(path=path, name=name))
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"{image_path} not found")
        h, w, c = image.shape
        anno = ET.parse(Path(self.anno_path.format(
            path=path, name=name))).getroot()  # 解析每一张图片的信息
        if self.target_transform is not None:
            ann = self.target_transform(anno)
        ann = np.array(
            ann).reshape(-1, 5) if len(ann) > 0 else (np.zeros((0, 5), dtype=np.float32))

        target = {
            'boxes': ann[:, :4].astype(np.float32),
            'labels': ann[:, 4].astype(np.int32),
            'origin_shape': [h, w]
        }

        return image, target


class Collate:
    def __call__(self, batch):

        targets, images = [], []
        '''将batch个照片和标注信息(类别和框) 打包成一个batch'''

        for sample in batch:
            image = sample[0]
            target = sample[1]
            images.append(image)
            targets.append(target)

        images = th.stack(images, dim=0)
        return images, targets


def build_datasets(basepath: str,
                   transform: Callable,
                   img_size: int,
                   is_train: bool
                   ):

    if is_train:
        img_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    else:
        img_sets = [('2007x', 'test')]
    dataset = VOCDection(basepath, img_size, img_sets=img_sets,
                         transform=transform)

    return dataset


def build_dataloader(
    dataset: VOCDection,
    batch_size: int,
    num_workers: int,
    collate: Callable = Collate(),
    pin_memory: bool = False,
):
    sample = RandomSampler(dataset)
    batch_sampler = BatchSampler(sample, batch_size, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


if __name__ == '__main__':
    from config.v1 import dataset_param, net_param
    from Net.v1.yolo import build_yolo

    is_train = True
    img_size = 480
    transform = build_transform(img_size, is_train)
    dataset = build_datasets(dataset_param.basepath, transform,
                             img_size, is_train=is_train)

    device = 'cuda'
    model = build_yolo(net_param, 'cuda', 0.001, 0.5, is_train)
    model.to(device)

    # p=np.random.randint(0,2000)
    # img,tgt=dataset[p]
    # ims=th.permute(img,(1,2,0)).contiguous().numpy()
    # ims=np.astype(ims,np.uint8).copy()

    # boxes=tgt['boxes']
    # labels=tgt['labels']

    # for i,boxe in enumerate(boxes):
    #     x1,y1,x2,y2=boxe
    #     cv2.rectangle(ims,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),1,4)
    #     text=VOC_CLASSES[labels[i]]
    #     cv2.putText(ims,VOC_CLASSES[labels[i]],(int(x1),int(y1)+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    # cv2.imshow('ims',ims)
    # cv2.waitKey(0)

    loader = build_dataloader(dataset, 64, 8)

    for images, targets in loader:
        # print(images.shape,targets)
        # break
        images = images.to(device)/255

        pred = model(images)
        print(pred)
        break
