## Net/v2 实现说明（与原版 YOLOv2 比较）

概述：这是一个手写、可读性为主的 YOLOv2 风格检测实现，包含 backbone、neck、head、matcher 与 loss。下文重点说明实现要点，并与原版 YOLOv2 的设计差异与作者做出的改进。

**一、重要实现思路**
- Backbone: 使用 DarkNet19 风格的卷积块 `Conv_Bn_LeakRelu`（Conv+BN+LeakyReLU）与 MaxPool 层堆叠，最终输出 1024 通道特征用于后续检测模块。
- Neck: 通过 1x1 降通道后，多次 MaxPool（固定 pool_size、stride=1）得到不同感受野特征并按通道拼接，再用 1x1 升维融合；实现轻量、低开销的多尺度信息聚合。
- Head: 分类与回归采用独立的卷积序列（`num_cls_head` / `num_reg_head` 可配置），分类通道数设为 `max(num_class,out_channels)`，回归支路输出定位特征供 `pred_boxes` 回归。
- Anchor 与网格：在 `make_grid` 中为特征图每个位置复制 k 个 anchor（来自 `config/v2/v2.yaml`），生成 (gx,gy,aw,ah) 组合用于解码。
- 解码与后处理：`decode_boxes` 用 sigmoid 预测中心偏移并加上 grid（并乘 stride），宽高用 exp(pred)*anchor；`postprocess` 用 sqrt(sigmoid(obj)*sigmoid(cls)) 计算得分，top-k 筛选后调用多类 NMS。
- Matcher 与 Loss：matcher 基于 anchor 与 GT 的 IoU 做正样本分配（若无超过阈值则取最大 IoU），loss 包括 obj（BCE）、cls（BCE，正样本乘以 IoU 权重）与 box（1-GIoU），按配置权重加和。

**二、与原版 YOLOv2 的差异与改进**

- Anchor 与尺度处理：
	- 差异：原版 YOLOv2 在特征图尺度上使用 anchor（通常在训练时以相对于特征图或输入尺寸的方式处理），并在解码时注意尺度变换。本实现将 anchor 尺寸直接以像素值放在 config 中并在 `make_grid` 中扩展到每个 grid，注释指出在解码宽高时无需再乘 stride（需保证训练/推理输入尺寸与 anchor 设计一致）。

- 特征融合（Neck）：
	- 差异：YOLOv2 原版常使用 passthrough（reorg）或更复杂的特征融合以保留高分辨率的信息（如使用更细粒度的浅层特征拼接）。本实现用多次池化拼接（低复杂度）代替。
	- 影响与改进：实现简单且内存/计算友好，便于学习与快速迭代；但在小目标/高分辨率细节方面，可能不如原版的 passthrough 融合效果。

- 后处理与得分合成：
	- 差异：原版常直接使用 objectness * class_score，而本实现用 sqrt(sigmoid(obj)*sigmoid(cls)) 平衡两者。 
	- 影响与改进：该策略在经验上能避免某一分支数值极端值主导排序，可能提升 top-k 筛选稳定性，但需在不同数据集上验证。

- 匹配与损失细节：
	- 差异：原版在 anchor 分配、负样本忽略区等策略上有多种实现，本实现以固定 IoU 阈值分配并在无匹配时选最大 IoU 作为正样本；分类损失对正样本乘以预测 IoU 做软权重，这一点是有意的改进。 
	- 影响与改进：IoU 加权分类标签可以弱化定位质量差的正样本对分类学习的负面影响，从而可能提高 NMS 后的最终质量（尤其在边界框质量参差时）。

- 模块化与可读性：
	- 差异：本实现偏学习与可读性，模块间接口清晰；原版实现为追求性能可能在细节上更复杂或更底层优化。 
	- 影响与改进：更容易替换 backbone/neck/loss 进行 ablation 或快速试验。

**三、训练入口与命令示例**
- 训练入口：仓库根目录的 `train.py`。构建 v2 模型由 `building/build_train.py` 中的 `build_net('v2', args)` 完成。

示例（CPU）：
```bash
python train.py --device cpu --img_size 640 --batch_size 16 --optim sgd --sche linear --epochs 150 --nms 0.5 --conf 0.005
```

示例（CUDA + AMP + 多尺度）：
```bash
python train.py --device cuda --img_size 640 --batch_size 32 --optim sgd --sche cosine --epochs 150 --nms 0.5 --conf 0.005 -ms -amp
```

断点恢复示例：
```bash
python train.py --device cuda --img_size 640 --batch_size 32 --resume
```

参数说明与注意事项：
- `--device`: `cpu`/`cuda`/`mps`。
- `--img_size`：网络输入（应与 stride/anchor 设计匹配）。
- `-ms`：多尺度训练；`-amp`：CUDA 下启用混合精度与梯度缩放。

**四、相关文件位置**
- `Net/v2/backbone.py`, `Net/v2/neck.py`, `Net/v2/head.py`, `Net/v2/yolo.py`
- `Match/v2/matcher.py`, `Match/v2/loss.py` 
- `config/v2/v2.yaml`（超参：anchors、head 层数、loss 权重）

如需我把 README 中某段展开（例如给出 matcher 的数值示例、或将 postprocess 的 top-k 筛选步骤可视化），我可以继续补充并写入该文件。

