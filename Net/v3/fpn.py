import torch as th
from torch import nn 
from torch.nn import functional as f

from Net.v3.basic import Conv,ConvBlock

class FPN(nn.Module):
    def __init__(self,in_dims=[256,512,1024],
                      out_dim:None|int=None,
                      width:int=1):
        
        super().__init__()

        self.in_dims=in_dims
        self.out_dim=out_dim

        c3,c4,c5=in_dims

        self.horiz_layer1=ConvBlock(c5,int(512*width))
        self.reduce_layer1=Conv(int(512*width),int(256*width),1,1,0)

        self.horiz_layer2=ConvBlock(c4+int(256*width),int(256*width))
        self.reduce_layer2=Conv(int(256*width),int(128*width),1,1,0)

        self.horiz_layer3=ConvBlock(c3+int(128*width),int(128*width))

        if out_dim is not None:
            self.out_layers=nn.ModuleList([
                Conv(in_dim,out_dim,1,1,0) for in_dim in [int(128*width),int(256*width),int(512*width)]
            ])
            self.out_dims=[out_dim]*3
        else:
            self.out_layers=None
            self.out_dims=[int(128*width),int(256*width),int(512*width)]
    
    def forward(self,x):

        c3,c4,c5=x

        p5=self.horiz_layer1(c5)
        p5_up=f.interpolate(self.reduce_layer1(p5),scale_factor=2)

        p4=self.horiz_layer2(th.cat([p5_up,c4],dim=1))
        p4_up=f.interpolate(self.reduce_layer2(p4),scale_factor=2)

        p3=self.horiz_layer3(th.cat([p4_up,c3],dim=1))
        
        out_featrues=[p3,p4,p5]

        return out_featrues
    


if __name__ == '__main__':
    
    c3=th.rand(1,256,64,64)
    c4=th.rand(1,512,32,32)
    c5=th.rand(1,1024,16,16)

    net=FPN(width=2)
    res=net([c3,c4,c5])
    for v in res:
        print(v.shape)


