import torch as th
from torch import nn
from Net.v3.basic import Conv,ResBlock

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.out_dims=[256,512,1024]

        self.layer1=nn.Sequential(
            Conv(3,32,3,1,1),
            Conv(32,64,3,2,1),
            ResBlock(64,64,1))          # //2
        
        self.layer2=nn.Sequential(
            Conv(64,128,3,2,1),         # //4
            ResBlock(128,128,2))
        
        self.layer3=nn.Sequential(
            Conv(128,256,3,2,1),        # //8
            ResBlock(256,256,8)
        )

        self.layer4=nn.Sequential(
            Conv(256,512,3,2,1),        #//16
            ResBlock(512,512,8)
        )

        self.layer5=nn.Sequential(
            Conv(512,1024,3,2,1),       #//32
            ResBlock(1024,1024,4)
        )

    def forward(self,x):

        c1=self.layer1(x)
        c2=self.layer2(c1)
        c3=self.layer3(c2) # 256,h//8
        c4=self.layer4(c3) # 512 h//16
        c5=self.layer5(c4) # 1024 h//32

        return [c3,c4,c5]
    
if __name__ == '__main__':
    
    net=Darknet53()

    data=th.rand(1,3,512,512)

    lis=net(data)

    for var in lis:
        print(var.shape)

    


        