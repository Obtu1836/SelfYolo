import torch as th
from torch import nn
from thop import profile


def compute(model:nn.Module,img_size:int,device:str):
    x=th.rand(1,3,img_size,img_size).to(device)
    print('='*20)
    flops,params=profile(model,inputs=(x,),verbose=False)[:2]
    print(f'GFLOPS: {flops/1e9*2:.2f}')
    print(f'Params: {params/1e6:.2f} M')
    print('='*20)

