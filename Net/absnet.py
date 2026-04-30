from abc import ABC,abstractmethod
from torch import nn

class YOLO(nn.Module,ABC):

    is_train:bool=True
    @abstractmethod
    def forward(self,x):...

    @abstractmethod
    def interface(self,x):...

