"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor
    vuv_accuracy: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.vuv_accuracy = detach_cpu(self.vuv_accuracy)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            phoneme_ids_list=batch.phoneme_ids_list,
            phoneme_durations_list=batch.phoneme_durations_list,
            phoneme_stress_list=batch.phoneme_stress_list,
            vowel_index_list=batch.vowel_index_list,
            speaker_id=batch.speaker_id,
        )

        # 予測結果とターゲットを結合して一括計算
        pred_f0_all = torch.cat(output_result.f0, dim=0)  # (sum(vL),)
        pred_vuv_all = torch.cat(output_result.vuv, dim=0)  # (sum(vL),)
        target_f0_all = torch.cat(batch.vowel_f0_means_list, dim=0)  # (sum(vL),)
        target_vuv_all = torch.cat(batch.vowel_voiced_list, dim=0)  # (sum(vL),)

        # vuv損失（全母音で計算）
        vuv_loss = binary_cross_entropy_with_logits(
            pred_vuv_all, target_vuv_all.float()
        )

        # F0損失（有声母音のみで計算）
        voiced_mask = target_vuv_all  # (sum(vL),)
        if voiced_mask.any():
            f0_loss = l1_loss(pred_f0_all[voiced_mask], target_f0_all[voiced_mask])
        else:
            f0_loss = pred_f0_all.new_tensor(0.0)

        loss = f0_loss + vuv_loss

        # 有声かどうかの精度
        pred_vuv_binary = pred_vuv_all > 0.0
        vuv_accuracy = (pred_vuv_binary == target_vuv_all).float().mean()

        return EvaluatorOutput(
            loss=loss,
            vuv_accuracy=vuv_accuracy,
            data_num=batch.data_num,
        )
