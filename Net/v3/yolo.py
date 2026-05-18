import torch as th 
from torch import nn 

from .backbone import Darknet53
from .fpn import FPN
from .neck import SPPF
from .head import Head

from config.v3 import cfg,logger

class Yolo(nn.Module):
    def __init__(self,cfg,
                 conf_thresh:float,
                 nms_thresh:float,
                 topk:int,
                 is_train:bool):
        super().__init__()

        self.cfg=cfg
        self.topk=topk
        self.nms_thresh=nms_thresh
        self.conf_thresh=conf_thresh
        self.stride=[8,16,32]

        self.is_train=is_train

        self.num_level=3
        self.num_anchors=len(cfg['net']['anchor_size'])//self.num_level
        self.anchor_size=th.as_tensor(cfg['net']['anchor_size']).float().\
                                      view(self.num_level,self.num_anchors,-1)
        self.backbone=Darknet53()
        out_dims=self.backbone.out_dims
        self.neck=SPPF(out_dims[-1],out_dims[-1],)
        self.fpn=FPN(out_dims,width=cfg['net']['width'])

        self.head_dims=self.fpn.out_dims

        self.heads=nn.ModuleList([Head(cfg,head_dim,head_dim) for \
                                  head_dim in self.head_dims])
        
        p=[nn.Conv2d(head.reg_out_dim,20,3,1,1) for head in self.heads]
        # self.pred_obj=nn.ModuleList([nn.Conv2d(head.reg_out_dim,1*self.num_anchors,1,1,0,bias=True) for head in self.heads] )
    
    def forward(self,x:th.Tensor):
        if not self.is_train:
            pass
        else:
            bs=x.shape[0]

            pyramid_feats=self.backbone(x) # [256 ,512,1024]
            pyramid_feats[-1]=self.neck(pyramid_feats[-1])
            pyramid_feats=self.fpn(pyramid_feats)#[128,256,512]

            fmp_sizes=[]
            pred_obj=[]
            pred_cls=[]
            pred_boxes=[]

            for level,(feat,head) in enumerate(zip(pyramid_feats,self.heads)):
                pass
            
if __name__ == '__main__':
    net=Yolo(cfg,0.05,0.2,100,True)
    data=th.rand(1,3,416,416)

    net(data)


        


