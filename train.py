import argparse
import time
import random
from pathlib import Path
from typing import Callable, Optional
from copy import deepcopy

import torch as th
from torch.nn import functional as f

from utils.dataset import build_datasets, build_dataloader, Collate
from utils.transforms import build_transform
from tools.optim import build_optimizer, build_lr_scheduler
from tools.compute_flops import compute
from eval import build_eval

from config.v1 import net_param, dataset_param, DatasetParam

from building.build_train import build_net

from Net.absnet import YOLO
class Trainer:
    def __init__(
                    self,
                    dataset_param: DatasetParam,
                    device: str,
                    img_size: int,
                    model:YOLO,
                    criterion: Callable,
                    optimizer_name: str,
                    lr_scheduler_name: str,
                    batch_size: int,
                    amp: bool,
                    mul_scale: bool,
                    stride: int,
                    epoches: int,
                    resume: bool = False,
                    resume_state: Optional[dict] = None,
                ):
        self.device = th.device(device)
        self.device_type = self.device.type
        self.best_map = float(resume_state.get('map', -1)
                              ) if resume_state is not None else -1
        self.epoches = epoches
        self.mul_scale = mul_scale
        self.stride = stride
        self.grad_thresh = 10
        self.criterion = criterion
        self.accumulate=1

        model_copy = deepcopy(model)
        model_copy.is_train = False
        model_copy.eval()
        compute(model_copy, img_size, device)
        del model_copy

        # 当device =cuda 且amp为True时 启用混合精度+梯度缩放策略
        self.amp_enabled = amp and self.device_type == "cuda"
        self.autocast_dtype: Optional[th.dtype] = th.float16 if self.amp_enabled else None
        self.use_grad_scaler = self.amp_enabled
        if self.amp_enabled:
            print('使用混合精度+梯度缩放')
            self.accumulate = max(1, round(64 / batch_size))
            print(f"Grad accumulate {self.accumulate}")
        else:
            print(f"{self.device_type.upper()} 不启用混合精度和梯度缩放，已回退到 FP32。")
        self.scaler: Optional[th.amp.GradScaler] = None  # type: ignore
        if self.device_type == "cuda":
            self.scaler = th.amp.GradScaler(  # type: ignore
                enabled=self.use_grad_scaler,#控制梯度缩放是否开启
            )

        # 构建预处理和数据集构造
        self.traintransform = build_transform(img_size, True)
        self.testtransform = build_transform(img_size, False)

        self.traindataset = build_datasets(
            dataset_param.basepath, self.traintransform, img_size, True
        )
        self.train_loader = build_dataloader(
            self.traindataset,
            batch_size,
            dataset_param.num_workers,
            Collate())

        # 构建优化器
        self.optimizer, self.scaler, self.start_epoch = build_optimizer(
            optimizer_name,
            model,
            resume=resume,
            grad_scaler=self.scaler,
            resume_state=resume_state,
        )
        # 构建学习率调度器
        self.lr_scheduler = build_lr_scheduler(
            lr_scheduler_name,
            self.optimizer,
            epoches,
            start_epoch=self.start_epoch,
            resume_state=resume_state,
        )

        # print('resume start_epoch:', self.start_epoch)
        # print('optimizer lr:', [g['lr'] for g in self.optimizer.param_groups])
        # print('lr_scheduler last_epoch:', getattr(self.lr_scheduler, 'last_epoch', None))
        # print('scaler restored:', self.scaler is not None)

        self.evaluator = build_eval(
            dataset_param.basepath, self.testtransform, device)

        

    def train(self, model:YOLO):
        if self.mul_scale:
            print("启用多尺度训练")
        for epoch in range(self.start_epoch, self.epoches):
            self.cur_epoch = epoch
            self.train_one_epoch(model)
            self.lr_scheduler.step()
            if (self.cur_epoch+1) % 5 == 0:
                self.eval(model)

    def train_one_epoch(self, model: YOLO):
        model.train()
        model.is_train = True
        epoch_size = len(self.train_loader)
        t0 = time.time()
        # 这一步的梯度清零兜底作用 比如继续训练时 读取的状态可能存在未清零
        self.optimizer.zero_grad()

        for iter_i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device).float() / 255.0
            shape = images.shape[-1]
            if self.mul_scale:  # 多尺度训练
                images, targets, shape = self.rescale_image_target(
                    images, targets)
            else:
                targets = self.refine_targets(targets)

            with th.autocast(device_type=self.device_type,
                             dtype=self.autocast_dtype,
                             enabled=self.amp_enabled):

                preds = model(images)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict["total_loss"] / self.accumulate

            if self.use_grad_scaler:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (iter_i+1) % 10 == 0:
                log = self.log_train(iter_i, epoch_size, loss_dict, shape)
                print(log)

            should_step = ((iter_i + 1) % self.accumulate ==
                           0) or ((iter_i + 1) == epoch_size)
            if not should_step:
                continue

            if self.grad_thresh > 0:
                if self.use_grad_scaler:
                    assert self.scaler is not None
                    self.scaler.unscale_(self.optimizer)
                th.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.grad_thresh)

            if self.use_grad_scaler:
                assert self.scaler is not None
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        print(f"epoch done, cost: {time.time() - t0:.2f}s")

    def rescale_image_target(self,
                             image: th.Tensor,
                             target: dict,
                             min_box_size=4,
                             muti_scale_range=[0.5, 1.3]):
        '''
        多尺度训练 将一个batch内的照片 随机放大或者缩小一定的倍数
        需保证是网络最大下采样倍数的整数倍
        并根据缩放倍数 调整标注的坐标 
        最后将坐标范围小于阈值的框去除
        '''
        old_shape = image.shape[-1]
        a, b = muti_scale_range
        # 随机生成倍数
        new_shape = random.randrange(
            int(a*old_shape), int(b*old_shape)+self.stride, self.stride)
        if new_shape/old_shape != 1:
            image = f.interpolate(input=image,
                                  size=new_shape,
                                  mode='bilinear',
                                  align_corners=False)
        # 修改标注的坐标
        for tgt in target:
            boxes = tgt['boxes'].clone()
            labels = tgt['labels'].clone()
            boxes = th.clamp(boxes, 0, old_shape)
            boxes[:, :4] *= new_shape/old_shape
            delta_wh = boxes[:, 2:]-boxes[:, :2]
            min_size = th.min(delta_wh, dim=-1)[0]#取最短边用来和阈值比较
            ind = min_size >= min_box_size
            tgt['boxes'] = boxes[ind]
            tgt['labels'] = labels[ind]

        return image, target, new_shape

    def refine_targets(self, targets: dict, min_box_size=4):
        '''
        不进行多尺度训练 只舍弃掉 框的w,h小于阈值的
        '''
        for tgt in targets:
            boxes = tgt['boxes'].clone()
            labels = tgt['labels'].clone()
            tgt_boxes_wh = boxes[:, 2:]-boxes[:, :2]
            min_size = th.min(tgt_boxes_wh, dim=-1)[0]
            ind = min_size >= min_box_size
            tgt['boxes'] = boxes[ind]
            tgt['labels'] = labels[ind]

        return targets

    def log_train(self, iter_i, epoch_size, loss_dict: dict, new_shape: int):
        log = f'[Epoch: {self.cur_epoch+1}/{self.epoches}] '
        log += f'[Iter: {iter_i+1}/{epoch_size} ]'
        for k in loss_dict.keys():
            log += f'[{k}: {loss_dict[k]:.4f} ]'
        log += f'[Size:{new_shape}]'
        cur_lr=(self.lr_scheduler.get_last_lr()[0])
        log+=f" lr: {round(cur_lr,6)}" #type: ignore

        return log

    def eval(self, model: YOLO):
        print('Evaluation model ....')
        model.eval()
        model.is_train = False

        with th.no_grad():
            self.evaluator.evaluate(model)

        cur_map = self.evaluator.map
        if cur_map > self.best_map:
            self.best_map = cur_map
            print('Saving epoch', self.cur_epoch+1)
            weight_name = 'model_best.pth' #默认的
            checkpoint_dir = Path(f'checkpoint')
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir/weight_name
            th.save({'model': model.state_dict(),
                     'map': round(self.best_map, 3),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict(),
                     'scaler': self.scaler.state_dict() if self.scaler else None,
                     'epoch': self.cur_epoch}, checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--device', default='mps', type=str)
    parser.add_argument('--img_size', default=480, type=int)
    parser.add_argument('--optim', default='sgd', type=str,
                        choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--sche', default='linear', type=str,
                        choices=['linear', 'cosine'])
    parser.add_argument('--topk',default=1000,type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-ms', action='store_true', default=False)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--nms', default=0.5, type=float)
    parser.add_argument('--conf', default=0.005, type=float)
    parser.add_argument('-r', '--resume', action='store_true', default=False)
    parser.add_argument("-amp", action="store_true",
                        default=False, help="enable mixed precision")
    parser.add_argument('--version','-v',default='v2',type=str)

    args = parser.parse_args()

    device = args.device
    resume_state = None
  
    model,critertion=build_net(args)

    save_path = Path(r'checkpoint\{args.version}_best_model.pth')
    if args.resume:
        checkpoint = th.load(
            save_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        resume_state = checkpoint
    model.to(device)

    trainer = Trainer(
        dataset_param=dataset_param,
        device=args.device,
        img_size=args.img_size,
        model=model,
        criterion=critertion,
        optimizer_name=args.optim,
        lr_scheduler_name=args.sche,
        batch_size=args.batch_size,
        amp=args.amp,
        mul_scale=args.ms,
        stride=net_param.stride,
        epoches=args.epochs,
        resume=args.resume,
        resume_state=resume_state,
    )
    trainer.train(model)
