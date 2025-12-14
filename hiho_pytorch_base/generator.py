"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .data.statistics import DataStatistics
from .network.predictor import (
    Predictor,
    create_padding_mask,
    create_predictor,
    get_lengths,
    pad_tensor_list,
)

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    f0: list[Tensor]  # [(vL,)]
    vuv: list[Tensor]  # [(vL,)]


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            statistics = DataStatistics(
                f0_mean=state_dict["f0_mean"].cpu().numpy(),
                f0_std=state_dict["f0_std"].cpu().numpy(),
                vuv_mean=state_dict["vuv_mean"].cpu().numpy(),
                vuv_std=state_dict["vuv_std"].cpu().numpy(),
            )
            predictor = create_predictor(config.network, statistics=statistics)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def _denorm(
        self,
        f0_list: list[Tensor],  # [(vL,)]
        vuv_list: list[Tensor],  # [(vL,)]
        speaker_id: Tensor,  # (B,)
    ) -> GeneratorOutput:
        speaker_id = speaker_id.to(self.device).long()
        speaker_f0_mean = self.predictor.f0_mean[speaker_id]  # type: ignore
        speaker_f0_std = self.predictor.f0_std[speaker_id]  # type: ignore
        speaker_vuv_mean = self.predictor.vuv_mean[speaker_id]  # type: ignore
        speaker_vuv_std = self.predictor.vuv_std[speaker_id]  # type: ignore

        out_f0_list: list[Tensor] = []
        out_vuv_list: list[Tensor] = []
        for i, (f0, vuv) in enumerate(zip(f0_list, vuv_list, strict=True)):
            out_f0_list.append(f0 * speaker_f0_std[i] + speaker_f0_mean[i])
            out_vuv_list.append(vuv * speaker_vuv_std[i] + speaker_vuv_mean[i])
        return GeneratorOutput(f0=out_f0_list, vuv=out_vuv_list)

    @torch.no_grad()
    def forward(
        self,
        *,
        noise_f0_list: list[TensorLike],  # [(L,)]
        noise_vuv_list: list[TensorLike],  # [(L,)]
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        speaker_id: TensorLike,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        """生成経路で推論する"""
        # TODO: 合っていそうか確認する

        def _convert(
            data: TensorLike,
        ) -> Tensor:
            return to_tensor(data, self.device)

        noise_f0_list_tensor = [to_tensor(item, self.device) for item in noise_f0_list]
        noise_vuv_list_tensor = [
            to_tensor(item, self.device) for item in noise_vuv_list
        ]
        speaker_id_tensor = _convert(speaker_id)

        if self.config.model.flow_type == "rectified_flow":
            return self._generate_rectified_flow(
                noise_f0_list=noise_f0_list_tensor,
                noise_vuv_list=noise_vuv_list_tensor,
                phoneme_ids_list=phoneme_ids_list,
                phoneme_durations_list=phoneme_durations_list,
                phoneme_stress_list=phoneme_stress_list,
                vowel_index_list=vowel_index_list,
                speaker_id=speaker_id_tensor,
                step_num=step_num,
            )
        elif self.config.model.flow_type == "meanflow":
            return self._generate_meanflow(
                noise_f0_list=noise_f0_list_tensor,
                noise_vuv_list=noise_vuv_list_tensor,
                phoneme_ids_list=phoneme_ids_list,
                phoneme_durations_list=phoneme_durations_list,
                phoneme_stress_list=phoneme_stress_list,
                vowel_index_list=vowel_index_list,
                speaker_id=speaker_id_tensor,
                step_num=step_num,
            )
        else:
            assert_never(self.config.model.flow_type)

    def _overwrite_noise(
        self,
        f0_list: list[Tensor],  # [(L,)]
        vuv_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
    ) -> None:
        for f0, vuv, vowel_index in zip(
            f0_list, vuv_list, vowel_index_list, strict=True
        ):
            length = f0.shape[0]
            vowel_mask = torch.zeros(length, device=f0.device, dtype=torch.bool)
            vowel_mask[vowel_index] = True

            step_noise_f0 = torch.randn_like(f0)
            step_noise_vuv = torch.randn_like(vuv)

            f0[~vowel_mask] = step_noise_f0[~vowel_mask]
            vuv[~vowel_mask] = step_noise_vuv[~vowel_mask]

            unvoiced_vowel_mask = (vuv[vowel_index] <= 0.0).to(torch.bool)
            if unvoiced_vowel_mask.any():
                unvoiced_vowel_index = vowel_index[unvoiced_vowel_mask]
                f0[unvoiced_vowel_index] = step_noise_f0[unvoiced_vowel_index]

    def _generate_rectified_flow(
        self,
        *,
        noise_f0_list: list[Tensor],  # [(L,)]
        noise_vuv_list: list[Tensor],  # [(L,)]
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        speaker_id: Tensor,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        f0_list = [noise_f0.clone() for noise_f0 in noise_f0_list]
        vuv_list = [noise_vuv.clone() for noise_vuv in noise_vuv_list]

        lengths = get_lengths(phoneme_ids_list)  # (B,)
        mask = create_padding_mask(lengths)  # (B, 1, L)
        padded_phoneme_ids = pad_tensor_list(phoneme_ids_list)  # (B, L)
        padded_phoneme_durations = pad_tensor_list(phoneme_durations_list)  # (B, L)
        padded_phoneme_stress = pad_tensor_list(phoneme_stress_list)  # (B, L)

        t_array = torch.linspace(0, 1, steps=step_num + 1, device=self.device)[:-1]
        delta_t_step = 1.0 / step_num

        for i in range(step_num):
            t = t_array[i].expand(len(f0_list))
            h = torch.zeros_like(t)

            padded_input_f0 = pad_tensor_list([t.unsqueeze(-1) for t in f0_list])
            padded_input_vuv = pad_tensor_list([t.unsqueeze(-1) for t in vuv_list])

            padded_velocity_f0, padded_velocity_vuv = self.predictor(
                padded_phoneme_ids=padded_phoneme_ids,
                padded_phoneme_durations=padded_phoneme_durations,
                padded_phoneme_stress=padded_phoneme_stress,
                padded_input_f0=padded_input_f0,
                padded_input_vuv=padded_input_vuv,
                mask=mask,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )  # (B, L, 1), (B, L, 1)

            for batch_index, (f0, vuv, vowel_index) in enumerate(
                zip(f0_list, vuv_list, vowel_index_list, strict=True)
            ):
                v_f0 = padded_velocity_f0[batch_index, vowel_index, 0]
                v_vuv = padded_velocity_vuv[batch_index, vowel_index, 0]
                f0[vowel_index] += v_f0 * delta_t_step
                vuv[vowel_index] += v_vuv * delta_t_step

            self._overwrite_noise(f0_list, vuv_list, vowel_index_list)

        output_f0_list = [
            f0[vowel_index]
            for f0, vowel_index in zip(f0_list, vowel_index_list, strict=True)
        ]
        output_vuv_list = [
            vuv[vowel_index]
            for vuv, vowel_index in zip(vuv_list, vowel_index_list, strict=True)
        ]

        return self._denorm(output_f0_list, output_vuv_list, speaker_id)

    def _generate_meanflow(
        self,
        *,
        noise_f0_list: list[Tensor],  # [(L,)]
        noise_vuv_list: list[Tensor],  # [(L,)]
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        speaker_id: Tensor,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        if step_num == 1:
            lengths = get_lengths(phoneme_ids_list)  # (B,)
            mask = create_padding_mask(lengths)  # (B, 1, L)
            padded_phoneme_ids = pad_tensor_list(phoneme_ids_list)  # (B, L)
            padded_phoneme_durations = pad_tensor_list(phoneme_durations_list)  # (B, L)
            padded_phoneme_stress = pad_tensor_list(phoneme_stress_list)  # (B, L)

            t = torch.ones(len(noise_f0_list), device=self.device)
            h = t

            padded_input_f0 = pad_tensor_list([t.unsqueeze(-1) for t in noise_f0_list])
            padded_input_vuv = pad_tensor_list(
                [t.unsqueeze(-1) for t in noise_vuv_list]
            )

            padded_velocity_f0, padded_velocity_vuv = self.predictor(
                padded_phoneme_ids=padded_phoneme_ids,
                padded_phoneme_durations=padded_phoneme_durations,
                padded_phoneme_stress=padded_phoneme_stress,
                padded_input_f0=padded_input_f0,
                padded_input_vuv=padded_input_vuv,
                mask=mask,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )  # (B, L, 1), (B, L, 1)

            f0_list = [noise_f0.clone() for noise_f0 in noise_f0_list]
            vuv_list = [noise_vuv.clone() for noise_vuv in noise_vuv_list]

            for batch_index, (f0, vuv, vowel_index) in enumerate(
                zip(f0_list, vuv_list, vowel_index_list, strict=True)
            ):
                v_f0 = padded_velocity_f0[batch_index, vowel_index, 0]
                v_vuv = padded_velocity_vuv[batch_index, vowel_index, 0]
                f0[vowel_index] -= v_f0
                vuv[vowel_index] -= v_vuv

            self._overwrite_noise(f0_list, vuv_list, vowel_index_list)

            output_f0_list = [
                f0[vowel_index]
                for f0, vowel_index in zip(f0_list, vowel_index_list, strict=True)
            ]
            output_vuv_list = [
                vuv[vowel_index]
                for vuv, vowel_index in zip(vuv_list, vowel_index_list, strict=True)
            ]

            return self._denorm(output_f0_list, output_vuv_list, speaker_id)
        else:
            f0_list = [noise_f0.clone() for noise_f0 in noise_f0_list]
            vuv_list = [noise_vuv.clone() for noise_vuv in noise_vuv_list]

            lengths = get_lengths(phoneme_ids_list)  # (B,)
            mask = create_padding_mask(lengths)  # (B, 1, L)
            padded_phoneme_ids = pad_tensor_list(phoneme_ids_list)  # (B, L)
            padded_phoneme_durations = pad_tensor_list(phoneme_durations_list)  # (B, L)
            padded_phoneme_stress = pad_tensor_list(phoneme_stress_list)  # (B, L)

            t_array = torch.linspace(1, 0, steps=step_num + 1, device=self.device)
            delta_t_step = 1.0 / step_num

            for i in range(step_num):
                t_start = t_array[i]
                t_end = t_array[i + 1]
                t = t_start.expand(len(f0_list))
                h = (t_start - t_end).expand(len(f0_list))

                padded_input_f0 = pad_tensor_list([t.unsqueeze(-1) for t in f0_list])
                padded_input_vuv = pad_tensor_list([t.unsqueeze(-1) for t in vuv_list])

                padded_velocity_f0, padded_velocity_vuv = self.predictor(
                    padded_phoneme_ids=padded_phoneme_ids,
                    padded_phoneme_durations=padded_phoneme_durations,
                    padded_phoneme_stress=padded_phoneme_stress,
                    padded_input_f0=padded_input_f0,
                    padded_input_vuv=padded_input_vuv,
                    mask=mask,
                    t=t,
                    h=h,
                    speaker_id=speaker_id,
                )  # (B, L, 1), (B, L, 1)

                for batch_index, (f0, vuv, vowel_index) in enumerate(
                    zip(f0_list, vuv_list, vowel_index_list, strict=True)
                ):
                    v_f0 = padded_velocity_f0[batch_index, vowel_index, 0]
                    v_vuv = padded_velocity_vuv[batch_index, vowel_index, 0]
                    f0[vowel_index] -= v_f0 * delta_t_step
                    vuv[vowel_index] -= v_vuv * delta_t_step

                self._overwrite_noise(f0_list, vuv_list, vowel_index_list)

            output_f0_list = [
                f0[vowel_index]
                for f0, vowel_index in zip(f0_list, vowel_index_list, strict=True)
            ]
            output_vuv_list = [
                vuv[vowel_index]
                for vuv, vowel_index in zip(vuv_list, vowel_index_list, strict=True)
            ]

            return self._denorm(output_f0_list, output_vuv_list, speaker_id)
