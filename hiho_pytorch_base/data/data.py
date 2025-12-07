"""データ処理モジュール"""

from dataclasses import dataclass

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
    f0: Tensor  # (T,) F0
    volume: Tensor  # (T,) 音量
    vowel_f0_means: Tensor  # (vL,) 各母音のF0
    vowel_voiced: Tensor  # (vL,) 各母音が有声か
    vowel_index: Tensor  # (vL,) 音素列のなかで母音のインデックス
    speaker_id: Tensor


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


def preprocess(d: InputData, is_eval: bool) -> OutputData:
    """全ての変換・検証・配列化処理を統合"""
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

    # Tensor変換
    return OutputData(
        phoneme_id=torch.from_numpy(phoneme_ids).long(),
        phoneme_duration=torch.from_numpy(phoneme_durations).float(),
        phoneme_stress=torch.from_numpy(phoneme_stress).long(),
        f0=torch.from_numpy(f0).float(),
        volume=torch.from_numpy(volume).float(),
        vowel_f0_means=torch.from_numpy(vowel_f0_means).float(),
        vowel_voiced=torch.from_numpy(vowel_voiced).bool(),
        vowel_index=torch.from_numpy(vowel_index).long(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )
