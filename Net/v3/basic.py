import torch as th
from torch import nn

class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x*th.sigmoid(x)

class Conv(nn.Module):
    def __init__(self,in_channel,out_channel,k=1,s=1,p=0):
        super().__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,k,s,p,bias=False),
            nn.BatchNorm2d(out_channel),
            SiLU(),
        )
    def forward(self,x):

        return self.layer(x)
    
class Bottlenck(nn.Module):
    def __init__(self,in_channel,out_channel,radio:float=0.5,shortcut:bool=False):
        super().__init__()

        mid_channel=int(out_channel*radio)
        self.conv1=Conv(in_channel,mid_channel,1)
        self.conv2=Conv(mid_channel,out_channel,3,1,1)

        self.shortcut=shortcut and in_channel==out_channel
    
    def forward(self,x):
        out=self.conv2(self.conv1(x))
        return x+out if self.shortcut else out
    
class ResBlock(nn.Module):
    def __init__(self,in_dim,out_dim,nums:int):
        super().__init__()

        assert in_dim==out_dim
        self.layer=nn.Sequential(
            *[Bottlenck(in_dim,out_dim,radio=0.5,shortcut=True) for _ in range(nums)])
    
    def forward(self,x):
        return self.layer(x)
    
class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()

        mid_dim=out_dim//2
        self.layer=nn.Sequential(
            Conv(in_dim,out_dim,1),
            Conv(out_dim,mid_dim,3,1,1),
            Conv(mid_dim,out_dim,1),
            Conv(out_dim,mid_dim,3,1,1),
            Conv(mid_dim,out_dim,1)
        )
    
    def forward(self,x):

        return self.layer(x)
    

        
    