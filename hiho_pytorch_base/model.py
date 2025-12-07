"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import Predictor
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    f0_loss: Tensor
    """F0損失"""

    vuv_loss: Tensor
    """vuv損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.f0_loss = detach_cpu(self.f0_loss)
        self.vuv_loss = detach_cpu(self.vuv_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        f0_output_list, vuv_output_list = self.predictor(
            phoneme_ids_list=batch.phoneme_ids_list,
            phoneme_durations_list=batch.phoneme_durations_list,
            phoneme_stress_list=batch.phoneme_stress_list,
            vowel_index_list=batch.vowel_index_list,
            speaker_id=batch.speaker_id,
        )  # [(vL,)], [(vL,)]

        # 一括で損失計算
        pred_f0_all = torch.cat(f0_output_list, dim=0)  # (sum(vL),)
        pred_vuv_all = torch.cat(vuv_output_list, dim=0)  # (sum(vL),)
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

        return ModelOutput(
            loss=loss,
            f0_loss=f0_loss,
            vuv_loss=vuv_loss,
            data_num=batch.data_num,
        )
