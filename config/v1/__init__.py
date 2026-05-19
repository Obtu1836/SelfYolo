import yaml
from dataclasses import dataclass, field

with open(r'config/v1/v1.yaml', 'r', encoding='utf-8') as f:
    v1config = yaml.safe_load(f)

with open(r'config\v1\.local.yaml', 'r', encoding='utf-8') as f:
    local_fig = yaml.safe_load(f)

v1config.update(local_fig)


@dataclass
class NetParam:
    num_cls_head: int
    num_reg_head: int
    stride: int
    num_class: int
    block_nums: list[int] = field(default_factory=list)
    loss_weight: list[float] = field(default_factory=list)


@dataclass
class DatasetParam:
    basepath: str
    num_workers: int
    detect_path: str


net = v1config['Net']
net_param = NetParam(num_cls_head=net['num_cls_head'],
                     num_reg_head=net['num_reg_head'],
                     stride=net['stride'],
                     block_nums=net['block_nums'],
                     num_class=net['num_class'],
                     loss_weight=net['loss_weight'])


dataset = v1config['dataset']
dataset_param = DatasetParam(
    basepath=dataset['basepath'],
    num_workers=dataset['num_workers'],
    detect_path=dataset['detect_path']
)



if __name__ == '__main__':
    print(dataset_param)
