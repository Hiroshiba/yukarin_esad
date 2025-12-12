"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    f0_mse_loss: Tensor
    vuv_accuracy: Tensor
    vuv_precision: Tensor
    vuv_recall: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.f0_mse_loss = detach_cpu(self.f0_mse_loss)
        self.vuv_accuracy = detach_cpu(self.vuv_accuracy)
        self.vuv_precision = detach_cpu(self.vuv_precision)
        self.vuv_recall = detach_cpu(self.vuv_recall)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.f0_mse_loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            noise_f0_list=batch.noise_f0_list,
            noise_vuv_list=batch.noise_vuv_list,
            phoneme_ids_list=batch.phoneme_ids_list,
            phoneme_durations_list=batch.phoneme_durations_list,
            phoneme_stress_list=batch.phoneme_stress_list,
            vowel_index_list=batch.vowel_index_list,
            speaker_id=batch.speaker_id,
            step_num=self.generator.config.train.diffusion_step_num,
        )

        # 予測結果とターゲットを結合して一括計算
        pred_f0_all = torch.cat(output_result.f0, dim=0)  # (sum(vL),)
        pred_vuv_all = torch.cat(output_result.vuv, dim=0)  # (sum(vL),)
        target_f0_all = torch.cat(batch.vowel_f0_means_list, dim=0)  # (sum(vL),)
        target_vuv_all = torch.cat(batch.vowel_voiced_list, dim=0)  # (sum(vL),)

        # F0損失（有声母音のみで計算）
        voiced_mask = target_vuv_all  # (sum(vL),)
        if voiced_mask.any():
            f0_mse = mse_loss(pred_f0_all[voiced_mask], target_f0_all[voiced_mask])
        else:
            f0_mse = pred_f0_all.new_tensor(0.0)

        # 有声かどうかの精度
        pred_vuv_binary = pred_vuv_all > 0.0
        tp = (pred_vuv_binary & target_vuv_all).float().sum()
        fp = (pred_vuv_binary & ~target_vuv_all).float().sum()
        fn = (~pred_vuv_binary & target_vuv_all).float().sum()
        vuv_precision = tp / (tp + fp)
        vuv_recall = tp / (tp + fn)
        vuv_accuracy = (pred_vuv_binary == target_vuv_all).float().mean()

        return EvaluatorOutput(
            f0_mse_loss=f0_mse,
            vuv_accuracy=vuv_accuracy,
            vuv_precision=vuv_precision,
            vuv_recall=vuv_recall,
            data_num=batch.data_num,
        )
