
yolo 手写系列 用于学习和理解yolo思想

✅ 支持的功能

📏 多尺度训练（支持说明）

- 目的：让模型在训练中见到不同分辨率，从而提升对尺度变化的鲁棒性。
- 做法（描述型示例）：在数据变换流程中随机选择若干候选尺寸并对输入与标注同时缩放

通过 train.py -ms 

⚡ 梯度缩放 / 混合精度（说明）

- 目的：在支持的硬件上使用混合精度（AMP）加速训练并节省显存，同时用 `GradScaler` 稳定梯度更新。

通过 train.py -amp --device cuda ...

🌍 支持优化器选择(sgd|adam)

- 目的：调参方便
通过 train.py --optim sgd|adam


🎶 版本选择
- 目的 本项目中支持3个yolo版本 分别基于v1,v2,v3并做了一定程度的改进
通过 train.py|detect.py -v v1|v2|v3 选择 

.... 其余简单功能 见train命令行参数


数据集构造：

本项目 使用的是voc2007+voc2012 其中训练部分使用的是2007trainval+2012trainval  验证集和测试集为2007 test 代码中已经完成分配,无需手动操作

1 数据集下载：
    https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012?resource=download
    下载并解压 并以解压后 以VOCdevkit路径为基准路径
    解压以后路径
        VOCdevkit
            ---VOC2007
            ---VOC2012

2 ✈️重要 需要在config/v1 目录下面 新建一个.local.yaml文件 ,
    (虽然代码中支持多个版本 但是路径是基于v1的 所以在v1下面新建就可以)
    格式如下：
    dataset:
        basepath: "xxxx/xxx/VOCdevkit" #xxx为自己的VOCdevkit的解压路径
        num_workers: 0 #自定义设置
        detect_path: "xxx/xxx/VOCdevkit/VOC2007/JPEGImages"
    
    以上格式 只需要变动xxx部分设置为自己的路径即可 其余部分不需要变动


