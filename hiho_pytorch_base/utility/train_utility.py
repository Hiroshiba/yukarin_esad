"""学習用ユーティリティ"""

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch


@dataclass
class DataNumProtocol:
    """data_numフィールドを持つdataclass"""

    data_num: int


@torch.no_grad()
def reduce_result[T: DataNumProtocol](results: list[T]) -> T:
    """結果をデータ数で重み付けして統計"""
    if not results:
        raise ValueError("results cannot be empty")

    result_dict: dict[str, Any] = {}
    sum_data_num = sum([r.data_num for r in results])

    for field in fields(results[0]):
        key = field.name
        if key == "data_num":
            continue

        values = [getattr(r, key) * r.data_num for r in results]
        if isinstance(values[0], torch.Tensor):
            stacked = torch.stack(values)
            if stacked.is_floating_point() and stacked.dtype != torch.float32:
                stacked = stacked.float()
            result_dict[key] = stacked.sum() / sum_data_num
        else:
            result_dict[key] = sum(values) / sum_data_num

    result_dict["data_num"] = sum_data_num

    result_class = type(results[0])
    return result_class(**result_dict)


def _flatten_dict(dd, separator="/", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in _flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


class Logger:
    """TensorBoardとW&Bを統合したロガー"""

    def __init__(
        self,
        config_dict: dict[str, Any],
        project_category: str | None,
        project_name: str,
        output_dir: Path,
    ):
        self.config_dict = config_dict
        self.project_category = project_category
        self.project_name = project_name
        self.output_dir = output_dir

        self.wandb_id = None

        self.wandb = None
        self.tensorboard = None

    def _initialize(self):
        import wandb
        import wandb.util
        from torch.utils.tensorboard import SummaryWriter

        if self.wandb_id is None:
            self.wandb_id = wandb.util.generate_id()

        self.wandb = wandb.init(
            id=self.wandb_id,
            project=self.project_category,
            name=self.project_name,
            dir=self.output_dir,
            resume="allow",
        )
        self.wandb.config.update(_flatten_dict(self.config_dict), allow_val_change=True)

        self.tensorboard = SummaryWriter(log_dir=self.output_dir)

    def log(self, summary: dict[str, Any], step: int):
        """ログ情報をTensorBoardとW&Bに送信"""
        if self.wandb is None or self.tensorboard is None:
            self._initialize()

        assert self.wandb is not None
        assert self.tensorboard is not None

        flattern_summary = _flatten_dict(summary)

        self.wandb.log(flattern_summary, step=step)

        for key, value in flattern_summary.items():
            self.tensorboard.add_scalar(key, value, step)

        print(f"Step: {step}, {flattern_summary}")

    def state_dict(self):
        """状態を取得"""
        state_dict = {"wandb_id": self.wandb_id}
        return state_dict

    def load_state_dict(self, state_dict):
        """状態を復元"""
        self.wandb_id = state_dict["wandb_id"]

    def close(self):
        """ロガーを閉じてリソースを開放"""
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
            self.tensorboard = None

        if self.wandb is not None:
            self.wandb.finish()
            self.wandb = None


class SaveManager:
    """ネットワークの保存を管理するクラス。最良と最新を保持する。"""

    def __init__(
        self,
        predictor: torch.nn.Module,
        prefix: str,
        output_dir: Path,
        top_num: int,
        last_num: int,
    ):
        self.predictor = predictor
        self.prefix = prefix
        self.output_dir = output_dir
        self.top_num = top_num
        self.last_num = last_num

        self.last_steps: list[int] = []
        self.top_step_values: list[tuple[int, float]] = []

    def save(self, value: float, step: int):
        """最良と最新を管理"""
        if not math.isfinite(value):
            return

        delete_steps: set[int] = set()
        judged = False

        # top N
        if (
            len(self.top_step_values) < self.top_num
            or value > self.top_step_values[-1][1]
        ):
            self.top_step_values.append((step, value))
            self.top_step_values.sort(key=lambda x: x[1], reverse=True)
            judged = True

        if len(self.top_step_values) > self.top_num:
            delete_steps.add(self.top_step_values.pop(-1)[0])

        # last N
        if len(self.last_steps) < self.last_num:
            self.last_steps.append(step)
            judged = True
        else:
            delete_steps.add(self.last_steps.pop(0))
            self.last_steps.append(step)
            judged = True

        # save and delete
        if judged:
            output_path = self.output_dir / f"{self.prefix}{step}.pth"
            tmp_output_path = self.output_dir / f"{self.prefix}{step}.pth.tmp"
            torch.save(self.predictor.state_dict(), tmp_output_path)
            tmp_output_path.rename(output_path)

        delete_steps = delete_steps - (
            set([x[0] for x in self.top_step_values]) | set(self.last_steps)
        )
        for delete_step in delete_steps:
            delete_path = self.output_dir / f"{self.prefix}{delete_step}.pth"
            if delete_path.exists():
                delete_path.unlink()

    def state_dict(self):
        """状態を取得"""
        state_dict = {
            "last_steps": self.last_steps,
            "top_step_values": self.top_step_values,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """状態を復元"""
        self.last_steps = state_dict["last_steps"]
        self.top_step_values = state_dict["top_step_values"]
