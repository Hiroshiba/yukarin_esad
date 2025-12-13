"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self, assert_never

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import (
    Predictor,
    create_padding_mask,
    get_lengths,
    pad_tensor_list,
)
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    f0_mse_loss: Tensor

    vuv_mse_loss: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.f0_mse_loss = detach_cpu(self.f0_mse_loss)
        self.vuv_mse_loss = detach_cpu(self.vuv_mse_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        if self.model_config.flow_type == "rectified_flow":
            return self._forward_rectified_flow(batch)
        elif self.model_config.flow_type == "meanflow":
            return self._forward_meanflow(batch)
        else:
            assert_never(self.model_config.flow_type)

    def _forward_rectified_flow(self, batch: BatchOutput) -> ModelOutput:
        """RectifiedFlowの損失を計算"""
        h = torch.zeros_like(batch.t)

        lengths = get_lengths(batch.phoneme_ids_list)  # (B,)
        mask = create_padding_mask(lengths)  # (B, 1, L)

        padded_phoneme_ids = pad_tensor_list(batch.phoneme_ids_list)  # (B, L)
        padded_phoneme_durations = pad_tensor_list(
            batch.phoneme_durations_list
        )  # (B, L)
        padded_phoneme_stress = pad_tensor_list(batch.phoneme_stress_list)  # (B, L)
        padded_input_f0 = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.input_f0_list]
        )  # (B, L, 1)
        padded_input_vuv = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.input_vuv_list]
        )  # (B, L, 1)

        padded_f0_v, padded_vuv_v = self.predictor(
            padded_phoneme_ids=padded_phoneme_ids,
            padded_phoneme_durations=padded_phoneme_durations,
            padded_phoneme_stress=padded_phoneme_stress,
            padded_input_f0=padded_input_f0,
            padded_input_vuv=padded_input_vuv,
            mask=mask,
            t=batch.t,
            h=h,
            speaker_id=batch.speaker_id,
        )  # (B, L, 1), (B, L, 1)

        target_vuv_v_list = [
            (target_vuv - noise_vuv)[vowel_index]
            for target_vuv, noise_vuv, vowel_index in zip(
                batch.target_vuv_list,
                batch.noise_vuv_list,
                batch.vowel_index_list,
                strict=True,
            )
        ]
        pred_vuv_v_all = torch.cat(
            [
                padded_vuv_v[i, vowel_index, 0]
                for i, vowel_index in enumerate(batch.vowel_index_list)
            ],
            dim=0,
        )  # (sum(vL),)
        tgt_vuv_v_all = torch.cat(target_vuv_v_list, dim=0)  # (sum(vL),)
        vuv_mse = mse_loss(pred_vuv_v_all, tgt_vuv_v_all)

        target_f0_v_list = [
            (target_f0 - noise_f0)[vowel_index]
            for target_f0, noise_f0, vowel_index in zip(
                batch.target_f0_list,
                batch.noise_f0_list,
                batch.vowel_index_list,
                strict=True,
            )
        ]
        pred_f0_v_all = torch.cat(
            [
                padded_f0_v[i, vowel_index, 0]
                for i, vowel_index in enumerate(batch.vowel_index_list)
            ],
            dim=0,
        )  # (sum(vL),)
        tgt_f0_v_all = torch.cat(target_f0_v_list, dim=0)  # (sum(vL),)
        target_voiced_all = torch.cat(batch.vowel_voiced_list, dim=0)  # (sum(vL),)

        if target_voiced_all.any():
            f0_mse = mse_loss(
                pred_f0_v_all[target_voiced_all], tgt_f0_v_all[target_voiced_all]
            )
        else:
            f0_mse = pred_f0_v_all.new_tensor(0.0)

        loss = f0_mse + vuv_mse

        return ModelOutput(
            loss=loss,
            f0_mse_loss=f0_mse,
            vuv_mse_loss=vuv_mse,
            data_num=batch.data_num,
        )

    def _forward_meanflow(self, batch: BatchOutput) -> ModelOutput:
        """MeanFlowの損失を計算"""
        lengths = get_lengths(batch.phoneme_ids_list)  # (B,)
        mask = create_padding_mask(lengths)  # (B, 1, L)

        mask_2d = mask.squeeze(1)  # (B, L)
        vowel_mask_2d = torch.zeros_like(mask_2d, dtype=torch.bool)  # (B, L)
        for i, vowel_index in enumerate(batch.vowel_index_list):
            vowel_mask_2d[i, vowel_index] = True

        voiced_mask_2d = torch.zeros_like(mask_2d, dtype=torch.bool)  # (B, L)
        for i, (vowel_index, vowel_voiced) in enumerate(
            zip(
                batch.vowel_index_list,
                batch.vowel_voiced_list,
                strict=True,
            )
        ):
            if vowel_voiced.any():
                voiced_mask_2d[i, vowel_index[vowel_voiced]] = True

        padded_phoneme_ids = pad_tensor_list(batch.phoneme_ids_list)  # (B, L)
        padded_phoneme_durations = pad_tensor_list(
            batch.phoneme_durations_list
        )  # (B, L)
        padded_phoneme_stress = pad_tensor_list(batch.phoneme_stress_list)  # (B, L)
        padded_input_f0 = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.input_f0_list]
        )  # (B, L, 1)
        padded_input_vuv = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.input_vuv_list]
        )  # (B, L, 1)
        padded_target_f0 = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.target_f0_list]
        )  # (B, L, 1)
        padded_noise_f0 = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.noise_f0_list]
        )  # (B, L, 1)
        padded_target_vuv = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.target_vuv_list]
        )  # (B, L, 1)
        padded_noise_vuv = pad_tensor_list(
            [t.unsqueeze(-1) for t in batch.noise_vuv_list]
        )  # (B, L, 1)

        # NOTE: JVP計算時にターゲットにNaNがあると結果がNaNになるためマスクする
        padded_target_f0_v = torch.where(
            voiced_mask_2d.unsqueeze(-1),
            padded_noise_f0 - padded_target_f0,
            torch.zeros_like(padded_noise_f0),
        )  # (B, L, 1)
        padded_target_vuv_v = torch.where(
            vowel_mask_2d.unsqueeze(-1),
            padded_noise_vuv - padded_target_vuv,
            torch.zeros_like(padded_noise_vuv),
        )  # (B, L, 1)

        def u_func(f0: Tensor, vuv: Tensor, t: Tensor, r: Tensor) -> Tensor:
            """JVP計算用のラッパー関数"""
            h = t - r
            f0_output, vuv_output = self.predictor(
                padded_phoneme_ids=padded_phoneme_ids,
                padded_phoneme_durations=padded_phoneme_durations,
                padded_phoneme_stress=padded_phoneme_stress,
                padded_input_f0=f0.unsqueeze(-1),
                padded_input_vuv=vuv.unsqueeze(-1),
                mask=mask,
                t=t,
                h=h,
                speaker_id=batch.speaker_id,
            )  # (B, L, 1), (B, L, 1)
            return torch.cat([f0_output, vuv_output], dim=2)  # (B, L, 2)

        jvp_result: tuple[Tensor, Tensor] = torch.func.jvp(
            func=u_func,
            primals=(
                padded_input_f0.squeeze(-1),
                padded_input_vuv.squeeze(-1),
                batch.t,
                batch.r,
            ),
            tangents=(
                padded_target_f0_v.squeeze(-1),
                padded_target_vuv_v.squeeze(-1),
                torch.ones_like(batch.t),
                torch.zeros_like(batch.r),
            ),
        )  # type: ignore
        u_pred, du_dt = jvp_result  # (B, L, 2), (B, L, 2)

        batch_size = batch.t.shape[0]
        max_length = padded_input_f0.size(1)
        h_expanded = (
            (batch.t - batch.r)
            .unsqueeze(1)
            .expand(batch_size, max_length)  # FIXME: .view()に変えられそう
        )  # (B, L)

        padded_target_v = torch.cat([padded_target_f0_v, padded_target_vuv_v], dim=2)
        u_tgt = padded_target_v - h_expanded.unsqueeze(-1) * du_dt  # (B, L, 2)
        mse_per_element = (u_pred - u_tgt.detach()) ** 2  # (B, L, 2)

        f0_mask = mask_2d & voiced_mask_2d  # (B, L)
        vuv_mask = mask_2d & vowel_mask_2d  # (B, L)

        masked_f0_mse = mse_per_element[:, :, 0] * f0_mask  # (B, L)
        masked_vuv_mse = mse_per_element[:, :, 1] * vuv_mask  # (B, L)

        f0_loss_per_sample = masked_f0_mse.sum(dim=1) / f0_mask.sum(dim=1)  # (B,)
        vuv_loss_per_sample = masked_vuv_mse.sum(dim=1) / vuv_mask.sum(dim=1)  # (B,)

        f0_mse = masked_f0_mse.sum() / f0_mask.sum()
        vuv_mse = masked_vuv_mse.sum() / vuv_mask.sum()

        loss_per_sample = f0_loss_per_sample + vuv_loss_per_sample  # (B,)
        adp_wt = (
            loss_per_sample.detach() + self.model_config.adaptive_weighting_eps
        ) ** self.model_config.adaptive_weighting_p  # (B,)
        loss_per_sample = loss_per_sample / adp_wt  # (B,)

        loss = loss_per_sample.mean()

        return ModelOutput(
            loss=loss,
            f0_mse_loss=f0_mse,
            vuv_mse_loss=vuv_mse,
            data_num=batch.data_num,
        )
