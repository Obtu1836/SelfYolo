import yaml
from dataclasses import dataclass, field

with open(r'config/v2/v2.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


@dataclass
class NetParam:
    num_cls_head: int
    num_reg_head: int
    num_class: int
    weights: list[float]=field(default_factory=list)
    anchor_size: list[list] = field(default_factory=list)


net = cfg['net']
net_param = NetParam(num_cls_head=net['num_cls_head'],
                     num_reg_head=net['num_reg_head'],
                     anchor_size=net['anchor_size'],
                     num_class=net['num_class'],
                     weights=net['weights'])

if __name__ == '__main__':
    import numpy as np
    print(type(net_param.anchor_size))
