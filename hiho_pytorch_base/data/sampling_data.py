"""時系列データとサンプリングレートを管理するクラス"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Self, assert_never

import librosa
import numpy
import scipy


class ResampleInterpolateKind(str, Enum):
    """リサンプリングの補間方法"""

    nearest = "nearest"
    linear = "linear"


class DegenerateType(str, Enum):
    """フレーム化した後のダウンサンプリング方法"""

    min = "min"
    max = "max"
    mean = "mean"
    median = "median"


@dataclass
class SamplingData:
    """時系列データとサンプリングレートを管理し、リサンプリング・分割・結合などを提供する"""

    array: numpy.ndarray  # shape: (N, ?)
    rate: float

    def resample(
        self,
        sampling_rate: float,
        index: int = 0,
        length: int | None = None,
        kind: ResampleInterpolateKind = ResampleInterpolateKind.nearest,
    ) -> numpy.ndarray:
        """リサンプリングしたデータを返す"""
        if length is None:
            length = int(len(self.array) / self.rate * sampling_rate)
        if kind == ResampleInterpolateKind.nearest:
            indexes = (
                numpy.random.default_rng().random() + index + numpy.arange(length)
            ) * (self.rate / sampling_rate)
            return self.array[indexes.astype(int)]
        else:
            indexes = (index + numpy.arange(length)) * (self.rate / sampling_rate)
            return scipy.interpolate.interp1d(
                numpy.arange(len(self.array)),
                self.array,
                kind=kind.value,
                axis=0,
                fill_value="extrapolate",
            )(indexes)

    def split(self, keypoint_seconds: list[float] | numpy.ndarray) -> list[Self]:
        """指定した秒数で分割する"""
        keypoint_seconds = numpy.array(keypoint_seconds)
        indexes = (keypoint_seconds * self.rate).astype(numpy.int32)
        arrays = numpy.split(self.array, indexes)
        return [self.__class__(array=array, rate=self.rate) for array in arrays]

    @classmethod
    def padding(cls, datas: list[Self], padding_value: numpy.ndarray) -> list[Self]:
        """パディングしてデータの長さを揃える"""
        datas = deepcopy(datas)

        max_length = max(len(d.array) for d in datas)
        for data in datas:
            padding_array = padding_value.repeat(max_length - len(data.array), axis=0)
            data.array = numpy.concatenate([data.array, padding_array])

        return datas

    def all_same(self) -> bool:
        """全てのデータが同じ値かどうかを確認する"""
        value = self.array[0][numpy.newaxis]
        return bool(numpy.all(value == self.array))

    @classmethod
    def collect(
        cls, datas: list[Self], rate: int, mode: str, error_time_length: float
    ) -> numpy.ndarray:
        """複数のデータを指定したサンプリングレートで結合する"""
        arrays: list[numpy.ndarray] = [
            d.resample(
                sampling_rate=rate, index=0, length=int(len(d.array) * rate / d.rate)
            )
            for d in datas
        ]

        # すべてのデータの長さが同じであることを確認する
        max_length = max(len(a) for a in arrays)
        for i, a in enumerate(arrays):
            if abs((max_length - len(a)) / rate) > error_time_length:
                raise ValueError(
                    f"データ{i}の長さが許容範囲を超えています: "
                    f"最大長{max_length / rate:.3f}秒に対し、データ長{len(a) / rate:.3f}秒 "
                    f"（許容誤差: {error_time_length}秒）"
                )

        if mode == "min":
            min_length = min(len(a) for a in arrays)
            array = numpy.concatenate([a[:min_length] for a in arrays], axis=1).astype(
                numpy.float32
            )

        elif mode == "max":
            arrays = [
                (
                    numpy.pad(a, ((0, max_length - len(a)), (0, 0)))
                    if len(a) < max_length
                    else a
                )
                for a in arrays
            ]
            array = numpy.concatenate(arrays, axis=1).astype(numpy.float32)

        elif mode == "first":
            first_length = len(arrays[0])
            arrays = [
                (
                    numpy.pad(a, ((0, first_length - len(a)), (0, 0)))
                    if len(a) < first_length
                    else a
                )
                for a in arrays
            ]
            array = numpy.concatenate(
                [a[:first_length] for a in arrays], axis=1
            ).astype(numpy.float32)

        else:
            raise ValueError(f"不明なモード: {mode}")

        return array

    def degenerate(
        self,
        frame_length: int,
        hop_length: int,
        centering: bool,
        padding_value: int | None,
        degenerate_type: DegenerateType,
    ) -> Self:
        """フレーム化してダウンサンプリングする"""
        array = self.array

        if centering:
            if padding_value is None:
                raise ValueError(
                    "centering=True のときは padding_value を指定してください"
                )
            width = [[frame_length // 2, frame_length // 2]] + [[0, 0]] * (
                array.ndim - 1
            )
            array = numpy.pad(array, width, constant_values=padding_value)

        array = numpy.ascontiguousarray(array)
        frame = librosa.util.frame(
            array, frame_length=frame_length, hop_length=hop_length, axis=0
        )

        match degenerate_type:
            case DegenerateType.min:
                array = numpy.min(frame, axis=1)
            case DegenerateType.max:
                array = numpy.max(frame, axis=1)
            case DegenerateType.mean:
                array = numpy.mean(frame, axis=1)
            case DegenerateType.median:
                array = numpy.median(frame, axis=1)
            case _:
                assert_never(degenerate_type)

        return type(self)(array=array, rate=self.rate / hop_length)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """読み込む"""
        d: dict = numpy.load(str(path), allow_pickle=True).item()
        array, rate = d["array"], d["rate"]

        if array.ndim == 1:
            array = array[:, numpy.newaxis]

        return cls(array=array, rate=rate)

    def save(self, path: PathLike):
        """保存する"""
        if self.array.ndim == 1:
            array = self.array[:, numpy.newaxis]
        else:
            array = self.array

        numpy.save(
            path, numpy.array(dict(array=array, rate=self.rate)), allow_pickle=True
        )
