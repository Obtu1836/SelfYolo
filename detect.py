import argparse
import cv2
import numpy as np
from numpy.typing import NDArray
import torch as th
import time
from pathlib import Path
from utils.dataset import VOC_CLASSES
from building.build_train import build_net
from config.v1 import dataset_param
from tools.get_device import get_device
import random
from utils.transforms import LetterResize
from config.v3 import logger


def preprocess_with_transform(img: np.ndarray, new_shape: int = 640,
                              stride: int = 32):
    transform = LetterResize(new_shape, True, stride=stride)
    padded, _, _, _ = transform(img, None, None)
    scale = min(new_shape / max(img.shape[:2]), 1.0)

    return padded, scale


def random_test_img():
    img_dir = Path(dataset_param.detect_path)
    files = [x for x in img_dir.iterdir() if x.is_file()
             and x.suffix.lower() == '.jpg']
    if len(files) == 0:
        raise FileNotFoundError(f'no jpg files in {img_dir}')
    img_path = random.sample(files, 1)[0]
    return img_path


def load_model(args, device,
               weight_path: Path,
               ):
    model, _ = build_net(args, device, is_train=False)
    model.to(device)
    check = th.load(weight_path, map_location=device, weights_only=False)
    state = check.get('model', None)
    if state is None:
        raise RuntimeError(
            f'No model state found in checkpoint: {weight_path}')
    model.load_state_dict(state)
    model.eval()
    return model


def drawing(img: NDArray, bboxes: NDArray, scores: NDArray, labels: NDArray):
    for i, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = VOC_CLASSES[int(label)]+f"{score:.2f}"
        cv2.putText(img, text, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 1, cv2.LINE_AA)

    return img


def main(args, device, image_path):
    weight_path = Path(
        r'checkpoint/{}_model_best.pth'.format(args.version))
    model = load_model(args, device, weight_path)

    img_path = image_path
    ori_img = cv2.imread(str(img_path))
    if ori_img is None:
        raise FileNotFoundError('img-path wrong')

    if args.shape == 0:
        origh_tmp, origw_tmp = ori_img.shape[:2]
        max_side = max(origh_tmp, origw_tmp)
        shape_for_letterbox = int(np.ceil(max_side / 32) * 32)
    else:
        shape_for_letterbox = args.shape

    padded, scale = preprocess_with_transform(
        ori_img, shape_for_letterbox, stride=32)
    img_tensor = th.from_numpy(padded).permute(
        2, 0, 1).contiguous().float()[None, ...].to(device) / 255.0
    t0 = time.time()
    with th.no_grad():
        bboxes, scores, labels = model(img_tensor)
    ts = round(time.time()-t0, 5)
    logger.info(f'cost time {ts} s')
    origh, origw = ori_img.shape[:2]

    if len(bboxes) > 0:
        bboxes = bboxes.copy()
        bboxes /= scale
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, origw)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, origh)

    ims = drawing(ori_img.copy(), bboxes, scores, labels)
    cv2.imshow('ims', ims)
    cv2.waitKey(0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--shape', default=640, type=int)
    parser.add_argument('--nms', default=0.5, type=float)
    parser.add_argument('--conf', default=0.35, type=float)
    parser.add_argument('--version', '-v', default='v2',
                        type=str, choices=['v1', 'v2'])
    parser.add_argument('--topk', default=200, type=int)

    args = parser.parse_args()
    device = get_device(args.device)

    # image_path = r'/Users/mac/program/VOCdevkit/VOC2007x/JPEGImages/000715.jpg'
    img_path = random_test_img()

    main(args, device, img_path)
