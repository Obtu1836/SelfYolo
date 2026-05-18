import yaml
from loguru import logger
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

LOG_DIR=ROOT/'LOG'
LOG_DIR.mkdir(exist_ok=True)

with open(r'config/v3/v3.yaml','r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f)

def config_logger():
    logger.remove()

    fmt= ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> |"
          "<level>{level: <4}</level> |"
          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |"
          "{message}"
        )
    
    logger.add(
        sys.stdout,
        format=fmt,
        # colorize=True,
        level="INFO",
        filter=lambda record:not record['extra'].get('loss',False))
    
    logger.add(LOG_DIR/'loss.log',
               level='INFO',
               format=fmt,
               filter=lambda record:record['extra'].get('loss',False))

config_logger()






