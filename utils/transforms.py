import cv2
import numpy as np
from numpy.typing import NDArray
import random

import torch as th


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    def __init__(self, series_transform):
        self.series_transform = series_transform

    def __call__(self, img, boxes, label):

        for transform in self.series_transform:
            img, boxes, label = transform(img, boxes, label)

        return img, boxes, label


class ConvertFromInts:
    def __call__(self, img: NDArray, boxes, label):
        return img.astype(np.float32), boxes, label


class RandomContrast():
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class ConvertColor():
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue():
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(0, 2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class Expand:
    def __call__(self, img, boxes, label):
        if random.randint(0, 2):
            return img, boxes, label
        h, w, c = img.shape
        radio = random.uniform(1, 4)
        left = int(random.uniform(0, w*radio-w))
        top = int(random.uniform(0, radio*h-h))

        bg = np.zeros((int(h*radio), int(w*radio), c), dtype=img.dtype)
        bg[top:top+h, left:left+w] = img

        img = bg

        boxes = boxes.copy()
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)

        return img, boxes, label


class RandomSampleCrop(object):

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes: NDArray, labels=NDArray):
        height, width, _ = image.shape
        # check
        if len(boxes) == 0:
            return image, boxes, labels

        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou or max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class RandomHorizontalFlip(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(0, 2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class Resize:
    def __init__(self, img_size=640):

        self.img_size = img_size

    def __call__(self, img, boxes, label):
        orih, oriw = img.shape[:2]
        image = cv2.resize(img, (self.img_size, self.img_size))

        boxes[:, [0, 2]] = boxes[:, [0, 2]]/oriw*self.img_size
        boxes[:, [1, 3]] = boxes[:, [1, 3]]/orih*self.img_size

        return image, boxes, label


class Augmentation:
    '''
    一些常见的数据增强
    '''
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.transfroms = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(),
            RandomSampleCrop(),                        # 随机剪裁
            RandomHorizontalFlip(),                    # 随机水平翻转
            Resize(self.img_size)
        ])

    def __call__(self, img: NDArray, target: dict):

        boxes = target['boxes'].copy()
        label = target['labels'].copy()
        image, boxes, label = self.transfroms(img, boxes, label)

        img_tensor = th.from_numpy(image).permute(2, 0, 1).contiguous().float()
        target['boxes'] = th.from_numpy(boxes).float()
        target['labels'] = th.from_numpy(label).long()

        return img_tensor, target


class Baseaugmention:
    '''
    验证和测试用 
    基础增强 只进行resize  target 参数需为None 之所以保留 是保持接口统一
    训练和验证时  需要传入target 测试时 target设置为None

    '''
    def __init__(self, img_size: int = 640):

        self.img_size = img_size

    def __call__(self, img: NDArray, target: None):
        orih, oriw = img.shape[:2]
        image = cv2.resize(img, (self.img_size, self.img_size))

        if target is not None:
            boxes = target['boxes'].copy()
            labels = target['labels'].copy()

            img_h, img_w = image.shape[:2]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] / oriw * img_w
            boxes[..., [1, 3]] = boxes[..., [1, 3]] / orih * img_h
            target['boxes'] = boxes

        img_tensor = th.from_numpy(image).permute(
            2, 0, 1).contiguous().float()
        if target is not None:
            target['boxes'] = th.from_numpy(boxes).float()
            target['labels'] = th.from_numpy(labels).long()

        return img_tensor, target


def build_transform(img_size: int, is_train: bool):

    if is_train:
        transform = Augmentation(img_size)
    else:
        transform = Baseaugmention(img_size)
    return transform
