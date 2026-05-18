import torch as th
from torch import nn 
from .basic import Conv

class SPPF(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,radio:float=0.5,
                 pool_ker:int=5):
        super().__init__()

        mid_dim=int(in_dim*radio)
        self.conv1=Conv(in_dim,mid_dim)
        self.conv2=Conv(mid_dim*4,out_dim)

        self.pool=nn.MaxPool2d(pool_ker,1,pool_ker//2)

    def forward(self,x):

        x=self.conv1(x)
        y1=self.pool(x)
        y2=self.pool(y1)
        y3=self.pool(y2)

        return self.conv2(th.cat([x,y1,y2,y3],dim=1))
    
if __name__ == '__main__':
    data=th.rand(1,3,32,32)
    net=SPPF(3,12)

    print(net(data).shape)
