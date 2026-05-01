from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pickle
import xml.etree.ElementTree as ET
import time
from typing import Callable
from utils.dataset import VOC_CLASSES, VOCDection
# from Net.v1.yolo import Yolo
from Net.absnet import YOLO
from config.v1 import dataset_param


class Evaluator:
    def __init__(self, path: str,
                 transform: Callable,
                 device: str,
                 img_size: int = 640):

        self.bashpath = Path(path)
        self.transform = transform
        self.device = device

        self.anno_path = lambda x: self.bashpath.joinpath('VOC2007x',
                                                          'Annotations', '{}.xml'.format(x))
        self.imgpath = lambda x: self.bashpath.joinpath('VOC2007x', 'JPEGImages',
                                                        '{}.jpg'.format(x))
        self.imgsetspath = self.bashpath.joinpath('VOC2007x', 'ImageSets',
                                                  'Main', 'test.txt')
        self.output_dir = Path('Evaloutput')
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = VOCDection(self.bashpath,
                                  img_size,
                                  transform=self.transform,
                                  img_sets=[('2007x', 'test')],
                                  )
        self.labelmap = VOC_CLASSES

    def evaluate(self, model: YOLO):
        bboxes = self.detect(model)
        self.record_detect(bboxes)
        del bboxes
        self.do_python_eval()

    def detect(self, model):
        num_images = len(self.dataset)
        # self.boxes 行是每个类别 列是每张图片 元素为每个标注框+类别
        all_boxes: list[list[NDArray]] = [[np.empty([0, 5], dtype=np.float32)
                                           for i in range(num_images)]
                                          for _ in range(len(self.labelmap))]

        for i in range(num_images):
            image = self.dataset.pull_image(i)
            h, w, _ = image.shape
            x, _ = self.transform(image, None)
            x = x[None, ...].to(self.device)/255

            t0 = time.time()
            boxes, scores, labels = model(x)
            detect_time = round(time.time()-t0, 5)
            origin_shape = [h, w]
            cur_shape = [*x.shape[-2:]]
            boxes = self.rescale_bboxes(boxes, origin_shape, cur_shape)

            for j in range(len(self.labelmap)):
                ind = np.where(labels == j)[0]
                if len(ind) == 0:
                    all_boxes[j][i] = np.empty([0, 5], np.float32)
                    continue
                c_boxes = boxes[ind]
                c_scores = scores[ind]
                x_det = np.hstack(
                    [c_boxes, c_scores[:, None]]).astype(np.float32)
                all_boxes[j][i] = x_det

            if (i+1) % 500 == 0:
                print(f'im_detect: {i+1}/{num_images} {detect_time:.5f}s')

        return all_boxes

    def record_detect(self, all_boxes):
        '''
        将all_boxes (检测信息)存储
        '''
        for cls_ind, cls in enumerate(self.labelmap):
            file_path = self.cls_file(cls)
            with open(file_path, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    # ids=[('self.basepath/2007x','000001'),(...)]
                    det_cls = all_boxes[cls_ind][im_ind]
                    if len(det_cls) == 0:
                        continue
                    for k in range(len(det_cls)):
                        f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n"
                                .format(index[1], det_cls[k, -1],
                                        det_cls[k, 0]+1, det_cls[k, 1]+1, det_cls[k, 2]+1, det_cls[k, 3]+1))
                        ''' 000001 置信度 x1,y1,x2,y2'''

    def cls_file(self, cls):

        file_name = 'det_'+'{}.txt'.format(cls)
        file_dir = Path.joinpath(self.output_dir, 'detect_cls')
        file_dir.mkdir(exist_ok=True)
        path = file_dir/file_name

        return path

    def do_python_eval(self, use_07: bool = True):

        cachedir = self.output_dir/'anno_cache'
        aps = []
        cachedir.mkdir(exist_ok=True)

        for i, cls in enumerate(self.labelmap):
            file_name = self.cls_file(cls)  # 按类别保存的预测信息
            rec,prec,ap=self.calculate_cls_ap(file_name, cls, cachedir,
                                  iou_thresh=0.5, use_07=use_07)
            aps+=[ap]
        
        self.map=np.mean(aps)

        print(f'Mean AP: {self.map:.4f}')

    def calculate_cls_ap(self, file_name: Path,
                         cls_name: str,
                         cachedir: Path,
                         iou_thresh=0.5,
                         use_07=True):
        '''
        读取test.txt的标注信息 并且存储  这个文件保存了所有用来测试图像的名称
        '''
        cachefile = cachedir/'annots.pkl'
        with open(self.imgsetspath, 'r') as r:
            lines = r.readlines()  # 读取用于测试的所有图像

        tgt_image_names = [x.strip() for x in lines]  # 获取图像名称 000001...
        if not cachefile.is_file():
            recs = {}
            for i, name in enumerate(tgt_image_names):
                recs[name] = self.parse_rec(
                    self.anno_path(name))  # 保存每一幅图像的标注信息

            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)

        else:
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        '''以上代码为 如果cachefile 不存在 则创建一个文件 该文件保存了所有图像的标注信息 
            如果文件存在 则直接读取'''

        class_recs = {}
        npos = 0
        for image_name in tgt_image_names:
            R = [obj for obj in recs[image_name] if obj['name'] == cls_name]  # 标注
            box = np.array([x['box'] for x in R])
            difficult = np.array([x['difficult'] for x in R], dtype=bool)
            det = [False]*len(R)
            npos = npos+sum(~difficult)
            class_recs[image_name] = {'box': box,
                                      'difficult': difficult,
                                      'det': det}
        '''
        以上代码文件 遍历每张图片 找出每张图片含有该类别的 返回含有该类别的图片标注信息 
        class_recs={'000001':{'box':[[x1,y1,x2,y3],[x11,y11,x22,y22]...],
                              'difficult':[0,1,...]...}}
        '''
        with open(file_name, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return np.array([]), np.array([]), 0.0

        splitlines = [x.strip().split(' ') for x in lines]
        image_name = [x[0] for x in splitlines]  # 图像的名称
        confidents = np.array([float(x[1]) for x in splitlines])  # 预测的置信度
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # 预测框

        sorted_ind = np.argsort(-confidents)
        BB = BB[sorted_ind, :]  # 将预测框按置信度排序
        image_name = [image_name[ind] for ind in sorted_ind]  # 名称也按图片排序 保持对应

        num = len(image_name)  # 所有的预测框
        tp = np.zeros(num) 
        fp = np.zeros(num)

        # 计算同一个类别下 同一张图片里 将预测框和标注框使用iou进行匹配
        for d in range(num):  # 遍历每一个预测框 使每一个预测框都去和标注框进行匹配(同一照片 同一类别)
            tgt = class_recs[image_name[d]]
            BBGT = tgt['box'].astype(np.float32)  # 标注框
            bb = BB[d, :].astype(np.float32)  # 预测框
            maxiou = -np.inf

            if BBGT.size > 0:
                x1 = np.maximum(bb[0], BBGT[:, 0])
                y1 = np.maximum(bb[1], BBGT[:, 1])
                x2 = np.minimum(bb[2], BBGT[:, 2])
                y2 = np.minimum(bb[3], BBGT[:, 3])

                iw = np.maximum(x2-x1, 0)
                ih = np.maximum(y2-y1, 0)
                inters = iw*ih
                union = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) +\
                    (BBGT[:, 2]-BBGT[:, 0]+1)*(BBGT[:, 3]-BBGT[:, 1]+1)-inters

                iou = inters/union
                maxiou = np.max(iou)
                ind = np.argmax(iou)
            '计算每一个预测框与标注框的iou'

            if maxiou > iou_thresh:
                if not tgt['difficult'][ind]:
                    if not tgt['det'][ind]:
                        tp[d] = 1
                        tgt['det'][ind] = 1
                    else:
                        fp[d] = 1
            else:
                fp[d] = 1
            '''以上代码意思为 如果预测框和标注框的iou都小于阈值 则该预测框fp=1 
            如果预测框匹配到已经匹配过的 同样说明该预测框错误 fp=1 
            如果能匹配到而且标注框不是困难的 则认为匹配成功 并标记标注框 防止别的预测框再匹配一次'''

        fp = np.cumsum(fp) 
        tp = np.cumsum(tp)

        if npos == 0:
            rec = np.zeros_like(tp)
            prec = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
            ap = 0.0
            return rec, prec, ap

        rec = tp/npos
        prec = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
        ap = self.calculate_ap(rec, prec, use_07)

        print(f'{cls_name} ap:: {round(ap, 5)}')
        return rec, prec, ap

    def calculate_ap(self, rec, prec, use_07):

        if use_07:
            ap = 0
            for i in np.arange(0, 1.1, 0.1):
                if np.sum(rec >= i) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= i])

                ap = ap+p/11
        else:
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [1.]))

            for i in range(mpre.size-1, 0, -1):
                mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum(mrec[i+1]-mrec[i])*mpre[i+1]

        return ap

    def parse_rec(self, image_path: Path):

        tree = ET.parse(image_path)
        objects = []

        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = parse_element(obj, 'name')
            obj_struct['difficult'] = int(
                parse_element(obj, 'difficult'))  # type: ignore
            bbox = obj.find('bndbox')
            if bbox is None:
                raise ValueError('anno wrong')

            x1 = parse_element(bbox, 'xmin')
            y1 = parse_element(bbox, 'ymin')
            x2 = parse_element(bbox, 'xmax')
            y2 = parse_element(bbox, 'ymax')
            obj_struct['box'] = [x1, y1, x2, y2]
            objects.append(obj_struct)
        return objects

    def rescale_bboxes(self, bboxes, origin_shape, cur_shape):

        orih, oriw = origin_shape
        curh, curw = cur_shape
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]]*oriw/curw
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]]*orih/curh

        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0, a_max=oriw)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0, a_max=orih)

        return bboxes


def parse_element(obj: ET.Element, tag: str):
    node = obj.find(tag)
    if node is None or node.text is None:
        raise ValueError('anno wrong')
    return int(node.text)


def build_eval(path,transform,device):
    return Evaluator(path,transform,device)


if __name__ == '__main__':

    path = dataset_param.basepath

    from utils.transforms import build_transform
    transform = build_transform(640, False)
    eval = Evaluator(path, transform, 'cuda')
    # p = eval.anno_path('000001')
    # pc = eval.parse_rec(p)
    # print(pc)
    from Net.v1.yolo import build_yolo
    from config.v1 import net_param
    device = 'cuda'
    model = build_yolo(net_param, device, 0.005, 0.5, False)
    model.to(device)
    eval.evaluate(model)
