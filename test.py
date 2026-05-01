import argparse
import cv2
import numpy as np
from numpy.typing import NDArray
import torch as th
import time
from pathlib import Path
from utils.dataset import VOC_CLASSES
from config.v2 import net_param
from tools.letterbox import letter
from Net.v2.yolo import build_yolo


def load_model(
        weight_path: Path,
        conf: float,
        nms: float,
        device: str,
):
    model = build_yolo(net_param, device, conf=conf,
                       nms=nms, topk=100,is_train=False)
    model.to(device)
    check = th.load(weight_path, map_location=device, weights_only=False)
    state = check.get('model', None)
    model.load_state_dict(state)
    model.eval()
    return model


def scale_coords_back(boxes: NDArray, scale, dh, dw, origh, origw):
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes *= scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origh)
    return boxes


def drawing(img: NDArray, bboxes: NDArray, scores: NDArray, labels: NDArray):
    for i, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = VOC_CLASSES[int(label)]+f"{score:.2f}"
        cv2.putText(img, text, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 1, cv2.LINE_AA)

    return img


def main(args, image_path):
    weight_path = Path(
        r'checkpoint\model_best.pth')
    model = load_model(weight_path, args.conf, args.nms, args.device)

    img_path = image_path
    ori_img = cv2.imread(img_path)
    if ori_img is None:
        raise FileNotFoundError('img-path wrong')
    letterimg, scale, halfdh, halfdw = letter(img_path, args.shape)
    img_tensor = th.from_numpy(letterimg[:, :, ::-1]/255).float()
    img_tensor = (img_tensor.permute(2, 0, 1)[None, ...]).to(args.device)

    t0 = time.time()
    with th.no_grad():
        bboxes, scores, labels = model(img_tensor)
    ts = round(time.time()-t0,5)
    print(f'cost time {ts} s')
    origh, origw = ori_img.shape[:2]
    if len(bboxes) > 0:
        bboxes = scale_coords_back(
            bboxes, scale, halfdh, halfdw, origh=origh, origw=origw)

    ims = drawing(ori_img, bboxes, scores, labels)

    ims = cv2.imshow('ims', ims)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--shape', default=640, type=int)
    parser.add_argument('--nms', default=0.5, type=float)
    parser.add_argument('--conf', default=0.5, type=float)

    args = parser.parse_args()

    image_path = r'D:\program\VOCdevkit\VOC2007x\JPEGImages\000013.jpg'

    main(args, image_path)
