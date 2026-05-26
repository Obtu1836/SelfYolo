## Net/v3 实现说明（与原版 YOLOv3 比较）

概述：这是一个手写、可读性优先的 YOLOv3 风格检测实现，整体由 Darknet53 backbone、SPPF、FPN、解耦 head、anchor matcher 与 loss 组成。实现目标不是完全复刻原版，而是在保留 YOLOv3 多尺度检测核心思路的基础上，做出更适合学习、调试和快速试验的模块化改写。

**一、重要实现思路**
- Backbone：使用 Darknet53 风格主干网络，位于 [Net/v3/backbone.py](Net/v3/backbone.py)。网络由 `Conv + BN + SiLU` 和残差块堆叠而成，输出三个尺度的特征图 `c3/c4/c5`，对应步长 `8/16/32`。
- Basic 模块：基础组件位于 [Net/v3/basic.py](Net/v3/basic.py)，包括 `Conv`、`Bottlenck`、`ResBlock` 和 `ConvBlock`。激活函数使用 `SiLU`，而不是 YOLOv3 原版常见的 LeakyReLU。
- Neck：在最高层特征进入 FPN 前，先通过 [Net/v3/neck.py](Net/v3/neck.py) 中的 `SPPF` 做多感受野聚合，再送入 [Net/v3/fpn.py](Net/v3/fpn.py) 的自顶向下 FPN。FPN 先逐层横向融合，再通过上采样把高层语义传到低层，实现三尺度检测特征构建。
- Head：检测头位于 [Net/v3/head.py](Net/v3/head.py)，分类和回归分支解耦，各自通过若干层卷积提取特征；最终在 [Net/v3/yolo.py](Net/v3/yolo.py) 中分别生成 `pred_obj`、`pred_cls` 和 `pred_bboxes` 三类输出。
- Anchor 与网格：anchor 配置在 [config/v3/v3.yaml](config/v3/v3.yaml) 中，以 `anchor_base_size=416` 作为参考尺寸存储。在 [Net/v3/yolo.py](Net/v3/yolo.py) 和 [Match/v3/matcher.py](Match/v3/matcher.py) 里，会按当前特征图对应的实际输入尺寸自动缩放 anchor 宽高，因此 anchor 可以从 416 基准迁移到 480 或多尺度训练。
- 解码与后处理：`pred_bboxes` 的前两维经过 `sigmoid` 后与网格中心相加，再乘对应 stride；宽高通过 `exp(pred) * anchor_wh` 解码为最终框。推理时使用 `sqrt(sigmoid(obj) * sigmoid(cls))` 作为排序分数，先做 top-k 过滤，再调用多类别 NMS。
- Matcher 与 Loss：匹配逻辑位于 [Match/v3/matcher.py](Match/v3/matcher.py)，对每个 GT 先与全部 anchor 计算 IoU，再把大于阈值的 anchor 都视为正样本；若都不满足，则回退到 IoU 最大的 anchor。损失定义位于 [Match/v3/loss.py](Match/v3/loss.py)，由 obj BCE、cls BCE 和 box GIoU 三部分组成，其中分类目标会乘以预测框 IoU 作为软权重。

**二、与原版 YOLOv3 的差异与改进**

- 激活函数与基础卷积块：
	- 差异：原版 YOLOv3 常用 `Conv + BN + LeakyReLU`，本实现统一使用 `Conv + BN + SiLU`。
	- 影响与改进：SiLU 在一些现代检测器中更常见，数值更平滑，通常对优化更友好；但这已经不再是严格意义上的原版 YOLOv3 复现。

- Neck 结构：
	- 差异：原版 YOLOv3 采用较直接的多尺度特征融合，本实现在最高层特征前额外加入了 `SPPF`，并用 [Net/v3/fpn.py](Net/v3/fpn.py) 中的 `ConvBlock` 做横向融合。
	- 影响与改进：SPPF 能以较低计算代价扩大感受野，增强高层语义；代价是结构比原版略有偏移，更接近后续 YOLO 系列的一些演化思路。

- 检测头形式：
	- 差异：原版 YOLOv3 更接近共享特征后直接输出检测张量，本实现采用解耦 head，把分类、回归、objectness 拆成不同分支。
	- 影响与改进：解耦 head 便于分别调整分类与定位容量，也更利于后续做 loss 或 head 结构消融；但与原版相比参数组织方式不同。

- Anchor 处理方式：
	- 差异：原版通常在固定训练尺寸下(416)组织 anchor，本实现把 anchor 存为基于 `anchor_base_size` 的参考宽高，再按当前输入尺度动态缩放。
	- 影响与改进：这让同一组 anchor 可以兼容 416、480 以及多尺度训练，工程上更灵活；但 anchor 是否最优仍取决于数据集目标尺寸分布。

- 正负样本分配：
	- 差异：原版 YOLOv3 常带有 ignore 策略，避免某些高 IoU 非最佳 anchor 被当成纯负样本。本实现采用“阈值内全收、否则取最大 IoU”的正样本策略，没有单独实现 ignore 区域。
	- 影响与改进：实现更直接、可读性更高，也更容易分析匹配结果；但训练中负样本约束可能更硬，导致 mAP 波动相对大一些。

- 分类监督与打分方式：
	- 差异：本实现的分类目标会乘以定位 IoU 作为软标签，推理得分使用 `sqrt(obj * cls)`，而不是简单相乘。
	- 影响与改进：这种做法有助于弱化低质量正样本对分类分支的影响，也能在排序时避免某一支路数值过于主导；是否优于传统方案仍需结合实验验证。

- Box 损失：
	- 差异：原版 YOLOv3 常见实现偏向坐标回归或 IoU 风格损失的变种，本实现直接使用 `1 - GIoU` 作为框损失。
	- 影响与改进：GIoU 对框几何关系更直接，通常比单纯回归坐标更稳定，也更符合当前检测训练的常见写法。

**三、训练入口与命令示例**
- 训练入口：仓库根目录的 [train.py](train.py)。构建 v3 模型由 [building/build_train.py](building/build_train.py) 中的 `build_net(args, device)` 完成。

示例（CPU）：
```bash
python train.py --device cpu --img_size 480 --batch_size 16 --optim sgd --sche linear --epochs 150 --nms 0.5 --conf 0.005 -v v3
```

示例（CUDA + AMP + 多尺度）：
```bash
python train.py --device cuda --img_size 480 --batch_size 32 --optim sgd --sche cosine --epochs 150 --nms 0.5 --conf 0.005 --topk 1000 -v v3 -ms -amp
```

断点恢复示例：
```bash
python train.py --device cuda --img_size 480 --batch_size 32 --resume -v v3
```

参数说明与注意事项：
- `--device`: `cpu` / `cuda` / `mps`。
- `--img_size`：网络输入尺寸；v3 中 anchor 会按当前尺寸相对 `anchor_base_size` 动态缩放。
- `--topk`：NMS 前保留的候选框数量。
- `-ms`：多尺度训练；`-amp`：仅在 CUDA 下启用混合精度与梯度缩放。
- `--conf`、`--nms`：分别控制置信度阈值和 NMS 阈值。

**四、相关文件位置**
- [Net/v3/basic.py](Net/v3/basic.py), [Net/v3/backbone.py](Net/v3/backbone.py), [Net/v3/neck.py](Net/v3/neck.py), [Net/v3/fpn.py](Net/v3/fpn.py), [Net/v3/head.py](Net/v3/head.py), [Net/v3/yolo.py](Net/v3/yolo.py)
- [Match/v3/matcher.py](Match/v3/matcher.py), [Match/v3/loss.py](Match/v3/loss.py)
- [config/v3/v3.yaml](config/v3/v3.yaml)（超参：anchors、head 层数、loss 权重、anchor_base_size）
