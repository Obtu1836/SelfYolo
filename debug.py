import torch as th
from Match.v2.matcher import Matcher
from Match.v2.loss import Critierion
from config.v2 import net_param
from utils.dataset import build_dataloader, build_datasets, build_transform
from config.v1 import dataset_param

transform = build_transform(640, True)
dataset = build_datasets(dataset_param.basepath, transform, 640, True)
loader = build_dataloader(dataset, 4, 0)
images, targets = next(iter(loader))

m = Matcher(net_param.num_class, 0.5, net_param.anchor_size)
fmp = th.tensor([640//32, 640//32])
gt_obj, gt_cls, gt_boxes = m(fmp, 32, targets)
print("pos anchors per image:", (gt_obj>0).sum(dim=(1,2,3,4)).numpy())