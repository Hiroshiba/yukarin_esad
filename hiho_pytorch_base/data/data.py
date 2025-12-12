"""データ処理モジュール"""

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy
import torch
from torch import Tensor

from .phoneme import ArpaPhoneme
from .sampling_data import ResampleInterpolateKind, SamplingData


@dataclass
class InputData:
    """データ処理前のデータ構造（SamplingData + ArpaPhonemeリストベース）"""

    phonemes: list[ArpaPhoneme]  # 音素のリスト（ストレス情報含む）
    f0_data: SamplingData  # F0のSamplingData
    volume_data: SamplingData  # volumeのSamplingData
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    phoneme_id: Tensor  # (L,) 音素ID
    phoneme_duration: Tensor  # (L,) 音素継続時間
    phoneme_stress: Tensor  # (L,) 全音素のストレス値（子音=0、母音=1-3）
    vowel_f0_means: Tensor  # (vL,) 各母音のF0
    vowel_voiced: Tensor  # (vL,) 各母音が有声か
    vowel_index: Tensor  # (vL,) 音素列のなかで母音のインデックス
    speaker_id: Tensor
    input_f0: Tensor  # (L,)
    target_f0: Tensor  # (L,)
    noise_f0: Tensor  # (L,)
    input_vuv: Tensor  # (L,)
    target_vuv: Tensor  # (L,)
    noise_vuv: Tensor  # (L,)
    t: Tensor
    r: Tensor


def calculate_vowel_f0_weighted_mean(
    f0: numpy.ndarray,
    volume: numpy.ndarray,
    vowel_index: numpy.ndarray,
    durations: numpy.ndarray,
    frame_rate: float,
) -> numpy.ndarray:
    """母音区間でのF0重み付け平均を計算する"""
    if len(vowel_index) == 0:
        raise ValueError(
            "母音インデックスが空です。LABファイルに母音が含まれていない可能性があります。"
        )

    # 音素の時間範囲を計算
    phoneme_times = numpy.cumsum(numpy.concatenate([[0], durations]))
    vowel_start_times = phoneme_times[vowel_index]
    vowel_end_times = phoneme_times[vowel_index + 1]
    vowel_start_frames = (vowel_start_times * frame_rate).astype(int)
    vowel_end_frames = (vowel_end_times * frame_rate).astype(int)

    # F0をNaNに変換（F0=0は無声区間）
    f0_masked = f0.copy().astype(float)
    f0_masked[f0_masked == 0] = numpy.nan

    # dB → 振幅変換
    volume_amplitude = numpy.power(10, volume / 20.0)

    # 各母音セグメントを処理
    vowel_f0_means = []
    for start_frame, end_frame in zip(
        vowel_start_frames, vowel_end_frames, strict=True
    ):
        f0_segment = f0_masked[start_frame:end_frame]
        volume_segment = volume_amplitude[start_frame:end_frame]

        # 有効なF0値のみで重み付け平均を計算
        valid_mask = ~numpy.isnan(f0_segment)

        if numpy.any(valid_mask) and numpy.sum(volume_segment[valid_mask]) > 0:
            weighted_mean = numpy.sum(
                f0_segment[valid_mask] * volume_segment[valid_mask]
            ) / numpy.sum(volume_segment[valid_mask])
            vowel_f0_means.append(weighted_mean)
        else:
            vowel_f0_means.append(0.0)

    return numpy.array(vowel_f0_means)


def sigmoid(a: float | numpy.ndarray) -> float | numpy.ndarray:
    """シグモイド関数"""
    return 1 / (1 + numpy.exp(-a))


def sample_time_meanflow(data_proportion: float) -> tuple[float, float]:
    """MeanFlow用の時間サンプリング (t, r)"""
    rng = numpy.random.default_rng()
    t_sample = float(sigmoid(rng.standard_normal() * 1.0 + (-0.4)))
    r_sample = float(sigmoid(rng.standard_normal() * 1.0 + (-0.4)))

    t = max(t_sample, r_sample)
    r = min(t_sample, r_sample)

    if rng.random() < data_proportion:
        r = t

    return t, r


def preprocess(
    d: InputData,
    is_eval: bool,
    flow_type: Literal["rectified_flow", "meanflow"],
    data_proportion: float,
) -> OutputData:
    """全ての変換・検証・配列化処理を統合"""
    rng = numpy.random.default_rng()

    # F0とボリュームのデータを取得
    f0 = d.f0_data.array
    volume = d.volume_data.array

    # リサンプリング
    frame_rate = d.f0_data.rate
    if abs(frame_rate - d.volume_data.rate) > 1e-4:
        volume = d.volume_data.resample(
            sampling_rate=frame_rate, index=0, kind=ResampleInterpolateKind.nearest
        )

    # F0と音量の整合性チェック
    # NOTE: 処理精度を考慮して3フレーム以内の誤差は許容する
    if abs(len(f0) - len(volume)) > 3:
        raise ValueError(
            f"F0と音量データの長さが一致しません:\n"
            f"  F0長:   {len(f0)}\n"
            f"  音量長: {len(volume)}\n"
            f"  許容範囲: 3フレーム以内"
        )

    # 長さを統一
    frame_length = min(len(f0), len(volume))
    f0 = f0[:frame_length]
    volume = volume[:frame_length]

    # 音素情報の抽出
    phoneme_ids = numpy.array(
        [ArpaPhoneme.phoneme_list.index(p.phoneme) for p in d.phonemes],
        dtype=numpy.int32,
    )
    phoneme_durations = numpy.array(
        [p.duration for p in d.phonemes], dtype=numpy.float32
    )

    # フレームレベルと音素レベルの整合性チェック
    # NOTE: 処理精度を考慮して3フレーム以内の誤差は許容する
    phoneme_duration = numpy.sum(phoneme_durations)
    phoneme_frame_length = int(phoneme_duration * frame_rate)
    if abs(frame_length - phoneme_frame_length) > 3:
        raise ValueError(
            f"LABファイルとフレーム数が一致しません:\n"
            f"  フレーム数:     {frame_length}\n"
            f"  音素フレーム数: {phoneme_frame_length}\n"
            f"  許容範囲:      3フレーム以内"
        )

    # 母音とそのストレス値を抽出
    vowel_indices = [
        i
        for i, phoneme in enumerate(d.phonemes)
        if ArpaPhoneme.is_vowel(phoneme.phoneme)
    ]

    # 全音素のストレス値を作成（子音=0、母音=1-3）
    phoneme_stresses = []
    for phoneme in d.phonemes:
        if ArpaPhoneme.is_vowel(phoneme.phoneme):
            if phoneme.stress is None:
                raise ValueError(
                    f"母音 '{phoneme.phoneme}' にストレス値が設定されていません"
                )
            stress_value = phoneme.stress + 1  # 0,1,2 -> 1,2,3
            phoneme_stresses.append(stress_value)
        else:
            phoneme_stresses.append(0)  # 子音は0

    vowel_index = numpy.array(vowel_indices)
    phoneme_stress = numpy.array(phoneme_stresses)

    # 母音ごとのF0重み付け平均を計算
    vowel_f0_means = calculate_vowel_f0_weighted_mean(
        f0=f0,
        volume=volume,
        vowel_index=vowel_index,
        durations=phoneme_durations,
        frame_rate=frame_rate,
    )

    # 有声か
    vowel_voiced = vowel_f0_means > 0

    # 全音素分のF0配列を作成（母音位置のみ実際の値、他はnan）
    phoneme_num = len(d.phonemes)
    phoneme_f0 = numpy.full(phoneme_num, numpy.nan, dtype=numpy.float64)
    phoneme_vuv = numpy.full(phoneme_num, numpy.nan, dtype=numpy.float64)

    for vowel_idx, f0_value, voiced_value in zip(
        vowel_index, vowel_f0_means, vowel_voiced, strict=True
    ):
        if voiced_value:
            phoneme_f0[vowel_idx] = f0_value
        phoneme_vuv[vowel_idx] = float(voiced_value)

    # Diffusion用の時間サンプリング
    match flow_type:
        case "meanflow":
            if is_eval:
                t, r = 1.0, 0.0
            else:
                t, r = sample_time_meanflow(data_proportion=data_proportion)
        case "rectified_flow":
            if is_eval:
                t, r = 0.0, 0.0
            else:
                t = float(sigmoid(rng.standard_normal()))
                r = 0.0
        case _:
            assert_never(flow_type)

    # F0のDiffusion処理
    target_f0 = phoneme_f0.copy()
    noise_f0 = rng.standard_normal(phoneme_num)

    match flow_type:
        case "meanflow":
            input_f0 = target_f0 + t * (noise_f0 - target_f0)
        case "rectified_flow":
            input_f0 = noise_f0 + t * (target_f0 - noise_f0)
        case _:
            assert_never(flow_type)

    voiced_mask = (phoneme_vuv == 1) & (~numpy.isnan(target_f0))
    input_f0 = numpy.where(voiced_mask, input_f0, noise_f0)

    # VUVのDiffusion処理
    target_vuv = phoneme_vuv.copy()
    noise_vuv = rng.standard_normal(phoneme_num)

    match flow_type:
        case "meanflow":
            input_vuv = target_vuv + t * (noise_vuv - target_vuv)
        case "rectified_flow":
            input_vuv = noise_vuv + t * (target_vuv - noise_vuv)
        case _:
            assert_never(flow_type)

    input_vuv = numpy.where(numpy.isnan(target_vuv), noise_vuv, input_vuv)

    # Tensor変換
    return OutputData(
        phoneme_id=torch.from_numpy(phoneme_ids).long(),
        phoneme_duration=torch.from_numpy(phoneme_durations).float(),
        phoneme_stress=torch.from_numpy(phoneme_stress).long(),
        vowel_f0_means=torch.from_numpy(vowel_f0_means).float(),
        vowel_voiced=torch.from_numpy(vowel_voiced).bool(),
        vowel_index=torch.from_numpy(vowel_index).long(),
        speaker_id=torch.tensor(d.speaker_id).long(),
        input_f0=torch.from_numpy(input_f0.astype(numpy.float32)).float(),
        target_f0=torch.from_numpy(target_f0.astype(numpy.float32)).float(),
        noise_f0=torch.from_numpy(noise_f0.astype(numpy.float32)).float(),
        input_vuv=torch.from_numpy(input_vuv.astype(numpy.float32)).float(),
        target_vuv=torch.from_numpy(target_vuv.astype(numpy.float32)).float(),
        noise_vuv=torch.from_numpy(noise_vuv.astype(numpy.float32)).float(),
        t=torch.tensor(t, dtype=torch.float32),
        r=torch.tensor(r, dtype=torch.float32),
    )
