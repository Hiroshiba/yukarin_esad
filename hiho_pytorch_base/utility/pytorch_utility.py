"""PyTorch関連ユーティリティ"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import is_dataclass
from typing import Any

import numpy
import torch
import torch_optimizer
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.optim.optimizer import Optimizer


def init_weights(model: torch.nn.Module, name: str) -> None:
    """指定された初期化手法で重みを初期化"""

    def _init_weights(layer: nn.Module):
        initializer: Callable
        if name == "uniform":
            initializer = torch.nn.init.uniform_
        elif name == "normal":
            initializer = torch.nn.init.normal_
        elif name == "xavier_uniform":
            initializer = torch.nn.init.xavier_uniform_
        elif name == "xavier_normal":
            initializer = torch.nn.init.xavier_normal_
        elif name == "kaiming_uniform":
            initializer = torch.nn.init.kaiming_uniform_
        elif name == "kaiming_normal":
            initializer = torch.nn.init.kaiming_normal_
        elif name == "orthogonal":
            initializer = torch.nn.init.orthogonal_
        elif name == "sparse":
            initializer = torch.nn.init.sparse_
        else:
            raise ValueError(name)

        for key, param in layer.named_parameters():
            if "weight" in key:
                try:
                    initializer(param)
                except Exception:
                    pass

    model.apply(_init_weights)


def make_optimizer(config_dict: dict[str, Any], model: nn.Module) -> Optimizer:
    """設定からオプティマイザーを作成"""
    from schedulefree import RAdamScheduleFree

    cp: dict[str, Any] = deepcopy(config_dict)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(), **cp)
    elif n == "ranger":
        optimizer = torch_optimizer.Ranger(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    elif n == "adamw":
        optimizer = optim.AdamW(model.parameters(), **cp)
    elif n == "true_adamw":
        cp["weight_decay"] /= cp["lr"]
        optimizer = optim.AdamW(model.parameters(), **cp)
    elif n == "radamschedulefree":
        optimizer = RAdamScheduleFree(model.parameters(), **cp)
    else:
        raise ValueError(n)

    return optimizer


class WarmupLR(LRScheduler):
    """ウォームアップ対応の学習率スケジューラー"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """ウォームアップを考慮した学習率を計算"""
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]


def make_scheduler(config_dict: dict[str, Any], optimizer: Optimizer) -> LRScheduler:
    """設定からスケジューラーを作成"""
    cp: dict[str, Any] = deepcopy(config_dict)
    n = cp.pop("name").lower()

    scheduler: LRScheduler
    if n == "step":
        scheduler = StepLR(optimizer, **cp)
    elif n == "warmup":
        scheduler = WarmupLR(optimizer, **cp)
    else:
        raise ValueError(n)

    return scheduler


def detach_cpu(data: Any) -> Any:
    """TensorをdetachしてCPUに移動、再帰的に処理"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, numpy.ndarray):
        return torch.as_tensor(data)
    elif isinstance(data, dict):
        try:
            return elem_type({key: detach_cpu(data[key]) for key in data})
        except TypeError:
            return {key: detach_cpu(data[key]) for key in data}
    elif isinstance(data, list | tuple):
        try:
            return elem_type([detach_cpu(d) for d in data])
        except TypeError:
            return [detach_cpu(d) for d in data]
    elif is_dataclass(data):
        raise TypeError(
            f"detach_cpuメソッドがないdataclassはサポートされていません: {type(data)}"
        )
    else:
        return data


def to_device(batch: Any, device: str, non_blocking: bool) -> Any:
    """データを指定されたデバイスに移動、再帰的に処理"""
    if isinstance(batch, dict):
        return {
            key: to_device(value, device, non_blocking) for key, value in batch.items()
        }
    elif isinstance(batch, list | tuple):
        return type(batch)(to_device(value, device, non_blocking) for value in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    elif is_dataclass(batch):
        raise TypeError(f"dataclassはサポートされていません: {type(batch)}")
    else:
        return batch
