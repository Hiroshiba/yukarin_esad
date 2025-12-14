"""データ処理関数のテスト"""

from typing import Literal

import numpy
import pytest
import torch

from hiho_pytorch_base.data.data import (
    InputData,
    OutputData,
    calculate_vowel_f0_weighted_mean,
    preprocess,
)
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.data.sampling_data import SamplingData
from hiho_pytorch_base.data.statistics import DataStatistics


def create_unit_statistics(speaker_id: int) -> DataStatistics:
    """単位統計情報を作成"""
    speaker_size = speaker_id + 1
    zeros = numpy.zeros(speaker_size, dtype=numpy.float64)
    ones = numpy.ones(speaker_size, dtype=numpy.float64)
    return DataStatistics(f0_mean=zeros, f0_std=ones, vuv_mean=zeros, vuv_std=ones)


def create_arpa_phonemes(
    phoneme_names: list[str],
    durations: list[float],
    stress_values: list[int | None],
) -> list[ArpaPhoneme]:
    """ArpaPhonemeのリストを作成"""
    phonemes = []
    start = 0.0
    for name, duration, stress in zip(
        phoneme_names, durations, stress_values, strict=True
    ):
        end = start + duration
        phoneme = ArpaPhoneme(phoneme=name, start=start, end=end, stress=stress)
        phonemes.append(phoneme)
        start = end

    return phonemes


def create_basic_input_data(
    phoneme_names: list[str],
    durations: list[float],
    stress_values: list[int | None],
    f0_values: numpy.ndarray,
    volume_amplitude: float,
    frame_rate: float,
    speaker_id: int,
) -> InputData:
    """テスト用の基本的なInputDataを作成"""
    phonemes = create_arpa_phonemes(phoneme_names, durations, stress_values)

    total_time = sum(durations)
    total_frames = int(total_time * frame_rate)

    volume_db = numpy.full(total_frames, 20 * numpy.log10(volume_amplitude))

    f0_data = SamplingData(array=f0_values[:, numpy.newaxis], rate=frame_rate)
    volume_data = SamplingData(array=volume_db[:, numpy.newaxis], rate=frame_rate)

    return InputData(
        phonemes=phonemes,
        f0_data=f0_data,
        volume_data=volume_data,
        speaker_id=speaker_id,
    )


def assert_output_data_types(output_data: OutputData) -> None:
    """OutputDataの型が正しいことを検証"""
    assert isinstance(output_data, OutputData)
    assert output_data.phoneme_id.dtype == torch.long
    assert output_data.phoneme_duration.dtype == torch.float
    assert output_data.vowel_f0_means.dtype == torch.float
    assert output_data.speaker_id.dtype == torch.long
    assert output_data.vowel_index.dtype == torch.long
    assert output_data.phoneme_stress.dtype == torch.long


def test_arpa_phoneme_classification():
    """アメリカ音素の母音・子音判定テスト"""
    for phoneme in ArpaPhoneme.vowel_phonemes:
        assert phoneme in ArpaPhoneme.phoneme_list
        assert ArpaPhoneme.is_vowel(phoneme)

    consonants = [
        p for p in ArpaPhoneme.phoneme_list if p not in ArpaPhoneme.vowel_phonemes
    ]
    for phoneme in consonants:
        assert not ArpaPhoneme.is_vowel(phoneme)


def test_new_input_data_structure(flow_type: Literal["meanflow", "rectified_flow"]):
    """新しいInputData構造の基本動作テスト"""
    input_data = create_basic_input_data(
        phoneme_names=["HH", "AA", "L"],
        durations=[0.1, 0.2, 0.1],
        stress_values=[None, 1, None],
        f0_values=numpy.ones(40) * 150.0,
        volume_amplitude=0.8,
        frame_rate=100.0,
        speaker_id=0,
    )

    # 基本構造の確認
    assert len(input_data.phonemes) == 3
    assert input_data.phonemes[0].phoneme == "HH"
    assert input_data.phonemes[1].phoneme == "AA"
    assert input_data.phonemes[2].phoneme == "L"

    assert input_data.phonemes[0].stress is None  # 子音
    assert input_data.phonemes[1].stress == 1  # 母音
    assert input_data.phonemes[2].stress is None  # 子音

    assert isinstance(input_data.f0_data, SamplingData)
    assert isinstance(input_data.volume_data, SamplingData)
    assert input_data.f0_data.rate == input_data.volume_data.rate

    output_data = preprocess(
        input_data,
        statistics=create_unit_statistics(input_data.speaker_id),
        is_eval=True,
        flow_type=flow_type,
        data_proportion=0.75,
    )
    assert_output_data_types(output_data)

    assert len(output_data.vowel_index) == 1
    assert len(output_data.phoneme_stress) == 3  # H AA T の3音素
    assert output_data.vowel_index[0] == 1  # AAの位置
    assert (
        output_data.phoneme_stress[output_data.vowel_index[0]] == 2
    )  # AAのストレス値（1+1=2）


def test_vowel_f0_weighted_mean_basic():
    """母音のF0重み付け平均計算の基本テスト"""
    durations = numpy.array([0.1, 0.2, 0.1, 0.2])
    vowel_index = numpy.array([1, 3])

    f0 = numpy.concatenate(
        [
            numpy.zeros(20),
            numpy.full(40, 150.0),  # 150Hz
            numpy.zeros(20),
            numpy.full(40, 180.0),  # 180Hz
        ]
    )
    volume_db = numpy.full(120, 20 * numpy.log10(0.8))

    result = calculate_vowel_f0_weighted_mean(
        f0, volume_db, vowel_index, durations, frame_rate=200.0
    )

    assert len(result) == 2
    numpy.testing.assert_almost_equal(result[0], 150.0, decimal=5)
    numpy.testing.assert_almost_equal(result[1], 180.0, decimal=5)


def test_f0_weighted_mean_with_partial_detection():
    """F0が部分的に検出失敗した場合の重み付け平均計算"""
    durations = numpy.array([0.1, 0.2])
    vowel_index = numpy.array([1])

    f0 = numpy.array([0, 0, 120, 130, 0, 140])
    volume_db = numpy.array([-10, -15, -5, -8, -12, -6])

    result = calculate_vowel_f0_weighted_mean(
        f0, volume_db, vowel_index, durations, frame_rate=20.0
    )

    assert len(result) == 1
    expected = (
        120 * 10 ** ((-5) / 20) + 130 * 10 ** ((-8) / 20) + 140 * 10 ** ((-6) / 20)
    ) / (10 ** ((-5) / 20) + 10 ** ((-8) / 20) + 10 ** ((-6) / 20))
    numpy.testing.assert_almost_equal(result[0], expected, decimal=1)


def test_preprocess_with_stress_phonemes(
    flow_type: Literal["meanflow", "rectified_flow"],
):
    """ストレス値付き音素のpreprocess動作テスト"""
    input_data = create_basic_input_data(
        phoneme_names=["S", "AA", "T"],
        durations=[0.1, 0.3, 0.1],
        stress_values=[None, 1, None],  # AAにストレス値1
        f0_values=numpy.concatenate(
            [
                numpy.zeros(10),  # S
                numpy.full(30, 160.0),  # AA1: 160Hz
                numpy.zeros(10),  # T
            ]
        ),
        volume_amplitude=0.8,
        frame_rate=100.0,
        speaker_id=0,
    )

    assert len(input_data.phonemes) == 3
    assert input_data.phonemes[1].phoneme == "AA"
    assert input_data.phonemes[1].stress == 1  # ストレス値確認

    output_data = preprocess(
        input_data,
        statistics=create_unit_statistics(input_data.speaker_id),
        is_eval=True,
        flow_type=flow_type,
        data_proportion=0.75,
    )
    assert len(output_data.vowel_f0_means) == 1
    assert output_data.vowel_index[0] == 1  # AAの位置
    assert (
        output_data.phoneme_stress[output_data.vowel_index[0]] == 2
    )  # ストレス値（1+1=2）


def test_time_mismatch_error(flow_type: Literal["meanflow", "rectified_flow"]):
    """時間不整合でエラーになることをテスト"""
    # 直接InputDataを作成して時間不整合を発生させる
    phonemes = create_arpa_phonemes(
        phoneme_names=["S", "AA", "T"],
        durations=[0.1, 0.1, 0.1],  # 合計0.3秒
        stress_values=[None, 1, None],
    )

    # F0データを意図的に長く設定（1.0秒相当）
    f0_data = SamplingData(array=numpy.zeros((100, 1)), rate=100.0)  # 1.0秒
    volume_data = SamplingData(array=numpy.full((100, 1), -20.0), rate=100.0)  # 1.0秒

    input_data = InputData(
        phonemes=phonemes,
        f0_data=f0_data,
        volume_data=volume_data,
        speaker_id=0,
    )

    with pytest.raises(ValueError, match="LABファイルとフレーム数が一致しません"):
        preprocess(
            input_data,
            statistics=create_unit_statistics(input_data.speaker_id),
            is_eval=True,
            flow_type=flow_type,
            data_proportion=0.75,
        )


def test_preprocess_basic_functionality(
    flow_type: Literal["meanflow", "rectified_flow"],
):
    """preprocess関数の基本動作テスト（新構造）"""
    durations = [0.02, 0.04, 0.03, 0.05, 0.02]  # 合計0.16秒
    frame_rate = 100.0
    total_frames = int(sum(durations) * frame_rate)  # 16フレーム

    input_data = create_basic_input_data(
        phoneme_names=["HH", "AA", "L", "AE", "T"],
        durations=durations,
        stress_values=[None, 1, None, 2, None],
        f0_values=numpy.ones(total_frames) * 150.0,
        volume_amplitude=0.8,
        frame_rate=frame_rate,
        speaker_id=42,
    )

    output_data = preprocess(
        input_data,
        statistics=create_unit_statistics(input_data.speaker_id),
        is_eval=True,
        flow_type=flow_type,
        data_proportion=0.75,
    )
    assert_output_data_types(output_data)

    # 母音が2つ（AA, AE）なので、vowel_indexの長さが2、phoneme_stressは全音素の5つ
    assert len(output_data.vowel_index) == 2
    assert len(output_data.phoneme_stress) == 5  # HH AA L AE T の5音素
    assert output_data.vowel_index[0] == 1  # AAの位置
    assert output_data.vowel_index[1] == 3  # AEの位置
    assert (
        output_data.phoneme_stress[output_data.vowel_index[0]] == 2
    )  # AAのストレス値（1+1=2）
    assert (
        output_data.phoneme_stress[output_data.vowel_index[1]] == 3
    )  # AEのストレス値（2+1=3）


def test_no_vowels_case(flow_type: Literal["meanflow", "rectified_flow"]):
    """母音が存在しない場合のテスト"""
    input_data = create_basic_input_data(
        phoneme_names=["T", "S", "K"],
        durations=[0.1, 0.1, 0.1],
        stress_values=[None, None, None],
        f0_values=numpy.zeros(30),
        volume_amplitude=0.03,
        frame_rate=100.0,
        speaker_id=0,
    )

    # 母音なしの場合はpreprocessでValueErrorが発生する
    with pytest.raises(ValueError, match="母音インデックスが空です"):
        preprocess(
            input_data,
            statistics=create_unit_statistics(input_data.speaker_id),
            is_eval=True,
            flow_type=flow_type,
            data_proportion=0.75,
        )


def test_multiple_stress_values_processing(
    flow_type: Literal["meanflow", "rectified_flow"],
):
    """複数の異なるストレス値（0,1,2）のpreprocess処理テスト"""
    input_data = create_basic_input_data(
        phoneme_names=["B", "AA", "T", "EH", "S"],
        durations=[0.1, 0.2, 0.1, 0.2, 0.1],
        stress_values=[None, 0, None, 2, None],  # AA0, EH2
        f0_values=numpy.concatenate(
            [
                numpy.zeros(10),  # B
                numpy.full(20, 120.0),  # AA0: 120Hz
                numpy.zeros(10),  # T
                numpy.full(20, 180.0),  # EH2: 180Hz
                numpy.zeros(10),  # S
            ]
        ),
        volume_amplitude=0.1,
        frame_rate=100.0,
        speaker_id=0,
    )

    # ストレス値が正しく設定されているか確認
    vowel_phonemes = [p for p in input_data.phonemes if ArpaPhoneme.is_vowel(p.phoneme)]
    assert len(vowel_phonemes) == 2  # AA0, EH2
    assert vowel_phonemes[0].phoneme == "AA"
    assert vowel_phonemes[0].stress == 0  # AA0のストレス値
    assert vowel_phonemes[1].phoneme == "EH"
    assert vowel_phonemes[1].stress == 2  # EH2のストレス値

    # preprocess後も保持されているか確認
    output_data = preprocess(
        input_data,
        statistics=create_unit_statistics(input_data.speaker_id),
        is_eval=True,
        flow_type=flow_type,
        data_proportion=0.75,
    )
    # 全音素のストレス値：B=0, AA0=1(0+1), T=0, EH2=3(2+1), S=0 → [0, 1, 0, 3, 0]
    assert torch.equal(output_data.phoneme_stress, torch.tensor([0, 1, 0, 3, 0]))
    assert torch.equal(output_data.vowel_index, torch.tensor([1, 3]))


def test_preprocess_frame_rate_mismatch(
    flow_type: Literal["meanflow", "rectified_flow"],
):
    """preprocess()で異なるフレームレートの処理テスト"""
    # F0とvolumeで異なるフレームレートを設定
    phonemes = create_arpa_phonemes(
        phoneme_names=["HH", "AA", "L"],
        durations=[0.1, 0.2, 0.1],
        stress_values=[None, 1, None],
    )

    # 異なるフレームレートのSamplingData
    f0_data = SamplingData(array=numpy.ones((80, 1)) * 150.0, rate=200.0)
    volume_data = SamplingData(array=numpy.ones((40, 1)) * (-20.0), rate=100.0)

    input_data = InputData(
        phonemes=phonemes, f0_data=f0_data, volume_data=volume_data, speaker_id=0
    )

    output_data = preprocess(
        input_data,
        statistics=create_unit_statistics(input_data.speaker_id),
        is_eval=True,
        flow_type=flow_type,
        data_proportion=0.75,
    )

    # リサンプリング後、母音F0が計算されることを確認
    assert len(output_data.vowel_f0_means) == 1
