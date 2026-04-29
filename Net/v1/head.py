import torch as th
from torch import nn

class Conv(nn.Module):
    def __init__(self,ins,ous,k,stride,p):
        super().__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(ins,ous,k,stride,p,bias=False),
            nn.BatchNorm2d(ous),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):

        return self.layer(x)
    

class Head(nn.Module):
    def __init__(self,in_channels:int,
                      num_class:int,
                      out_channels:int=512,
                      num_cls_head:int=2,
                      num_reg_head:int=2):
        super().__init__()

        cls_layer=[]
        cls_outdim=max(out_channels,num_class)
        for i in range(num_cls_head):
            if i==0:
                cls_layer.append(Conv(in_channels,cls_outdim,3,1,1))
            else:
                cls_layer.append(Conv(cls_outdim,cls_outdim,3,1,1))

        reg_layer=[]
        reg_outdim=cls_outdim
        for i in range(num_reg_head):
            if i==0:
                reg_layer.append(Conv(in_channels,reg_outdim,3,1,1))
            else:
                reg_layer.append(Conv(reg_outdim,reg_outdim,3,1,1))
        
        self.cls_layers=nn.Sequential(*cls_layer)
        self.reg_layers=nn.Sequential(*reg_layer)
    
    def forward(self,x):

        cls=self.cls_layers(x)
        reg=self.reg_layers(x)

        return cls,reg

if __name__ == '__main__':
    
    net=Head(512,20,512,2,2)
    data=th.rand(1,512,7,7)

    cls,reg=net(data)
    print(cls.shape,reg.shape)


