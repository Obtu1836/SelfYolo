import torch as th
from torch import nn 
from .basic import Conv

from config.v3 import cfg

class Head(nn.Module):
    def __init__(self,cfg,in_dim:int,out_dim:int):
        super().__init__()

        self.in_dim=in_dim
        self.num_cls_head=cfg['net']['num_cls_head']
        self.num_reg_head=cfg['net']['num_reg_head']

        num_class=cfg['net']['num_class']

        cls_feats=[]
        self.cls_out_dim:int=int(max(out_dim,num_class))
        for i in range(self.num_cls_head):
            if i==0:
                cls_feats.append(Conv(in_dim,out_dim,3,1,1))
            else:
                cls_feats.append(Conv(out_dim,out_dim,3,1,1))
        
        reg_feats=[]
        self.reg_out_dim:int=int(max(out_dim,64))
        for i in range(self.num_reg_head):
            if i==0:
                reg_feats.append(Conv(in_dim,out_dim,3,1,1))
            else:
                reg_feats.append(Conv(out_dim,out_dim,3,1,1))

        self.cls_feats=nn.Sequential(*cls_feats)
        self.reg_feats=nn.Sequential(*reg_feats)
    
    def forward(self,x):
        cls_feats=self.cls_feats(x)
        reg_feats=self.reg_feats(x)

        return cls_feats,reg_feats
    
if __name__ == '__main__':
    net=Head(cfg,256,256)
    data=th.rand(1,256,10,10)
    a,b=net(data)
    print(a.shape,b.shape)

