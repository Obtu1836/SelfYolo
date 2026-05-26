import yaml
from loguru import logger
import sys
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = ROOT/'LOG'
LOG_DIR.mkdir(exist_ok=True)

with open(r'config/v3/v3.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


def config_logger():
    logger.remove()

    fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> |"
           "<level>{level: <4}</level> |"
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |"
           "{message}"
           )

    logger.add(
        sys.stdout,
        format=fmt,
        # colorize=True,
        level="INFO",
        filter=lambda record: not record['extra'].get('loss', False))

    logger.add(LOG_DIR/'loss.log',
               level='INFO',
               format=fmt,
               filter=lambda record: record['extra'].get('loss', False))


config_logger()


@dataclass
class NetParam:
    num_class: int
    width: int
    num_cls_head: int
    num_reg_head: int
    num_anchors: int
    anchor_base_size: int
    weights: list[float] = field(default_factory=list)
    anchor_size: list[list[int]] = field(default_factory=list)


net = cfg['net']
net_param = NetParam(num_class=net['num_class'],
                     width=net['width'],
                     num_cls_head=net['num_cls_head'],
                     num_reg_head=net['num_reg_head'],
                     num_anchors=net['num_anchors'],
                     anchor_base_size=net['anchor_base_size'],
                     weights=net['weights'],
                     anchor_size=net['anchor_size'])
