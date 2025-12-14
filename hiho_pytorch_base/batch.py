"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from .data.data import OutputData
from .utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    phoneme_ids_list: list[Tensor]  # [(L,)]
    phoneme_durations_list: list[Tensor]  # [(L,)]
    phoneme_stress_list: list[Tensor]  # [(L,)]
    vowel_f0_means_list: list[Tensor]  # [(vL,)]
    vowel_voiced_list: list[Tensor]  # [(vL,)]
    vowel_index_list: list[Tensor]  # [(vL,)]
    speaker_id: Tensor  # (B,)
    input_f0_list: list[Tensor]  # [(L,)]
    target_f0_list: list[Tensor]  # [(L,)]
    noise_f0_list: list[Tensor]  # [(L,)]
    input_vuv_list: list[Tensor]  # [(L,)]
    target_vuv_list: list[Tensor]  # [(L,)]
    noise_vuv_list: list[Tensor]  # [(L,)]
    t: Tensor  # (B,)
    r: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.phoneme_ids_list = to_device(
            self.phoneme_ids_list, device, non_blocking=non_blocking
        )
        self.phoneme_durations_list = to_device(
            self.phoneme_durations_list, device, non_blocking=non_blocking
        )
        self.phoneme_stress_list = to_device(
            self.phoneme_stress_list, device, non_blocking=non_blocking
        )
        self.vowel_f0_means_list = to_device(
            self.vowel_f0_means_list, device, non_blocking=non_blocking
        )
        self.vowel_voiced_list = to_device(
            self.vowel_voiced_list, device, non_blocking=non_blocking
        )
        self.vowel_index_list = to_device(
            self.vowel_index_list, device, non_blocking=non_blocking
        )
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        self.input_f0_list = to_device(
            self.input_f0_list, device, non_blocking=non_blocking
        )
        self.target_f0_list = to_device(
            self.target_f0_list, device, non_blocking=non_blocking
        )
        self.noise_f0_list = to_device(
            self.noise_f0_list, device, non_blocking=non_blocking
        )
        self.input_vuv_list = to_device(
            self.input_vuv_list, device, non_blocking=non_blocking
        )
        self.target_vuv_list = to_device(
            self.target_vuv_list, device, non_blocking=non_blocking
        )
        self.noise_vuv_list = to_device(
            self.noise_vuv_list, device, non_blocking=non_blocking
        )
        self.t = to_device(self.t, device, non_blocking=non_blocking)
        self.r = to_device(self.r, device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        phoneme_ids_list=[d.phoneme_id for d in data_list],
        phoneme_durations_list=[d.phoneme_duration for d in data_list],
        phoneme_stress_list=[d.phoneme_stress for d in data_list],
        vowel_f0_means_list=[d.vowel_f0_means for d in data_list],
        vowel_voiced_list=[d.vowel_voiced for d in data_list],
        vowel_index_list=[d.vowel_index for d in data_list],
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
        input_f0_list=[d.input_f0 for d in data_list],
        target_f0_list=[d.target_f0 for d in data_list],
        noise_f0_list=[d.noise_f0 for d in data_list],
        input_vuv_list=[d.input_vuv for d in data_list],
        target_vuv_list=[d.target_vuv for d in data_list],
        noise_vuv_list=[d.noise_vuv for d in data_list],
        t=collate_stack([d.t for d in data_list]),
        r=collate_stack([d.r for d in data_list]),
    )
