import math
from typing import Optional

import torch as th
from torch import nn
from torch.optim import SGD, Adam


def build_optimizer(
    name: str,
    model: nn.Module,
    resume: bool,
    grad_scaler: Optional[th.amp.GradScaler],  # type: ignore
    resume_state: Optional[dict] = None,
):
    optim_dict = {
        "optimize": name,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "lr0": 0.01,
    }
    g = [], [], []

    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            if isinstance(v, bn):
                g[1].append(v.weight)
            else:
                g[0].append(v.weight)

    if optim_dict["optimize"] == "adam":
        optimizer = Adam(g[2], lr=optim_dict["lr0"])
        print(f'优化器: Adam')
    elif optim_dict["optimize"] == "sgd":
        optimizer = SGD(
            g[2],
            lr=optim_dict["lr0"],
            momentum=optim_dict["momentum"],
            nesterov=True,
        )
        print(f'优化器: SGD ')
    else:
        raise NotImplementedError("optimizer not implement")

    optimizer.add_param_group({"params": g[0], "weight_decay": optim_dict["weight_decay"]})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0})

    start_epoch = 0
    if resume and resume_state is not None:
        
        optimizer.load_state_dict(resume_state["optimizer"])
        
        gs = resume_state.get("scaler", None)
        if grad_scaler is not None and gs is not None:
            grad_scaler.load_state_dict(gs)

        start_epoch = int(resume_state.get("epoch", -1)) + 1
    return optimizer, grad_scaler, start_epoch


def build_lr_scheduler(
    name: str,
    optimizer: th.optim.Optimizer,
    max_epoch: int,
    start_epoch: int = 0,
    resume_state: Optional[dict] = None,
):
    scheduler_dict = {"scheduler": name, "lrf": 0.01}
    if scheduler_dict["scheduler"] == "cosine":
        def fun(x):
            return ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (
                scheduler_dict["lrf"] - 1) + 1
        
    elif scheduler_dict["scheduler"] == "linear":
        def fun(x):
            return (1 - x / max_epoch) * (1.0 - scheduler_dict["lrf"]) + scheduler_dict["lrf"]
    else:
        raise ValueError("lr_scheduler not callable")

    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=fun)

    # 恢复 scheduler 状态（若有）
    if resume_state is not None and resume_state.get("lr_scheduler") is not None:
        scheduler.load_state_dict(resume_state["lr_scheduler"])
        # 同步学习率 使optimizer.param_groups 与保存时 lr 一致
        if resume_state.get("optimizer") is not None:
            saved_pgs = resume_state["optimizer"].get("param_groups", [])
            for i, pg in enumerate(optimizer.param_groups):
                try:
                    pg["lr"] = saved_pgs[i]["lr"]
                except Exception:
                    pass

    elif start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    return scheduler
