"""統計情報モジュール"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Self

import numpy
from upath import UPath

from ..config import DataFileConfig, DatasetConfig
from ..data.sampling_data import SamplingData
from ..utility.upath_utility import to_local_path


@dataclass
class DataStatistics:
    """話者ごとの統計情報"""

    f0_mean: numpy.ndarray
    f0_std: numpy.ndarray
    vuv_mean: numpy.ndarray
    vuv_std: numpy.ndarray

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から統計情報を生成"""
        return cls(
            f0_mean=numpy.asarray(d["f0_mean"], dtype=numpy.float64),
            f0_std=numpy.asarray(d["f0_std"], dtype=numpy.float64),
            vuv_mean=numpy.asarray(d["vuv_mean"], dtype=numpy.float64),
            vuv_std=numpy.asarray(d["vuv_std"], dtype=numpy.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        """統計情報を辞書に変換"""
        return {
            "f0_mean": self.f0_mean.tolist(),
            "f0_std": self.f0_std.tolist(),
            "vuv_mean": self.vuv_mean.tolist(),
            "vuv_std": self.vuv_std.tolist(),
        }


def _get_statistics_cache_key_and_info(
    config: DataFileConfig,
) -> tuple[str, dict[str, str | None]]:
    root_dir = None if config.root_dir is None else str(config.root_dir)

    speaker_dict_text = config.speaker_dict_path.read_text()
    speaker_dict_hash = hashlib.sha256(
        speaker_dict_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    f0_pathlist_text = config.f0_pathlist_path.read_text()
    f0_pathlist_hash = hashlib.sha256(
        f0_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    info = {
        "root_dir": root_dir,
        "f0_pathlist_path": str(config.f0_pathlist_path),
        "f0_pathlist_hash": f0_pathlist_hash,
        "speaker_dict_path": str(config.speaker_dict_path),
        "speaker_dict_hash": speaker_dict_hash,
    }

    cache_key = hashlib.sha256(
        json.dumps(info, sort_keys=True, ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
    ).hexdigest()
    return cache_key, info


@dataclass(frozen=True)
class StatisticsDataInput:
    """統計情報計算用データ"""

    f0_path: UPath
    speaker_id: int


def _load_statistics_item(
    d: StatisticsDataInput,
) -> tuple[int, int, float, float, int, float, float]:
    f0_data = SamplingData.load(to_local_path(d.f0_path))
    f0 = f0_data.array.astype(numpy.float64).reshape(-1)

    voiced = f0 > 0.0
    f0_voiced = f0[voiced]

    f0_voiced_count = int(f0_voiced.size)
    f0_voiced_sum = float(f0_voiced.sum())
    f0_voiced_sumsq = float((f0_voiced * f0_voiced).sum())

    vuv = voiced.astype(numpy.float64)
    vuv_count = int(vuv.size)
    vuv_sum = float(vuv.sum())
    vuv_sumsq = float((vuv * vuv).sum())

    return (
        d.speaker_id,
        f0_voiced_count,
        f0_voiced_sum,
        f0_voiced_sumsq,
        vuv_count,
        vuv_sum,
        vuv_sumsq,
    )


def _calc_statistics(
    datas: list[StatisticsDataInput],
    *,
    workers: int,
) -> DataStatistics:
    """話者ごとの統計情報を取得"""
    if workers <= 0:
        raise ValueError(f"workers must be > 0: {workers}")
    if len(datas) == 0:
        raise ValueError("datas is empty")

    max_speaker_id = max(d.speaker_id for d in datas)
    speaker_size = max_speaker_id + 1

    f0_voiced_count = numpy.zeros(speaker_size, dtype=numpy.int64)
    f0_voiced_sum = numpy.zeros(speaker_size, dtype=numpy.float64)
    f0_voiced_sumsq = numpy.zeros(speaker_size, dtype=numpy.float64)

    vuv_count = numpy.zeros(speaker_size, dtype=numpy.int64)
    vuv_sum = numpy.zeros(speaker_size, dtype=numpy.float64)
    vuv_sumsq = numpy.zeros(speaker_size, dtype=numpy.float64)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_load_statistics_item, d) for d in datas]
        for future in as_completed(futures):
            (
                speaker_id,
                item_f0_voiced_count,
                item_f0_voiced_sum,
                item_f0_voiced_sumsq,
                item_vuv_count,
                item_vuv_sum,
                item_vuv_sumsq,
            ) = future.result()

            f0_voiced_count[speaker_id] += item_f0_voiced_count
            f0_voiced_sum[speaker_id] += item_f0_voiced_sum
            f0_voiced_sumsq[speaker_id] += item_f0_voiced_sumsq

            vuv_count[speaker_id] += item_vuv_count
            vuv_sum[speaker_id] += item_vuv_sum
            vuv_sumsq[speaker_id] += item_vuv_sumsq

    f0_mean = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)
    f0_std = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)
    vuv_mean = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)
    vuv_std = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)

    for speaker_id in range(speaker_size):
        if f0_voiced_count[speaker_id] > 0:
            mean = f0_voiced_sum[speaker_id] / f0_voiced_count[speaker_id]
            var = (
                f0_voiced_sumsq[speaker_id] / f0_voiced_count[speaker_id] - mean * mean
            )
            f0_mean[speaker_id] = mean
            f0_std[speaker_id] = numpy.sqrt(var)

        if vuv_count[speaker_id] > 0:
            mean = vuv_sum[speaker_id] / vuv_count[speaker_id]
            var = vuv_sumsq[speaker_id] / vuv_count[speaker_id] - mean * mean
            vuv_mean[speaker_id] = mean
            vuv_std[speaker_id] = numpy.sqrt(var)

    # NOTE: vuv_std が 0 の話者は他話者の中央値でフォールバックする
    valid_vuv_std = numpy.isfinite(vuv_std) & (vuv_std > 0.0)
    zero_vuv_std = vuv_std == 0.0
    if numpy.any(zero_vuv_std):
        print("vuv_std が 0 の話者が存在するため、他話者の中央値でフォールバックします")
        vuv_mean[zero_vuv_std] = numpy.median(vuv_mean[valid_vuv_std])
        vuv_std[zero_vuv_std] = numpy.median(vuv_std[valid_vuv_std])

    return DataStatistics(
        f0_mean=f0_mean, f0_std=f0_std, vuv_mean=vuv_mean, vuv_std=vuv_std
    )


def get_or_calc_statistics(
    config: DatasetConfig,
    datas: list[StatisticsDataInput],
    *,
    workers: int,
) -> DataStatistics:
    """統計情報を取得または計算する"""
    cache_key, info = _get_statistics_cache_key_and_info(config.train)
    cache_dir = config.statistics_cache_dir / cache_key
    info_path = cache_dir / "info.json"
    statistics_path = cache_dir / "statistics.json"

    if statistics_path.exists():
        print(f"統計情報をキャッシュから読み込みました: {statistics_path}")
        statistics_dict = json.loads(statistics_path.read_text())
        return DataStatistics.from_dict(statistics_dict)

    print(f"統計情報を計算しています... (データ数: {len(datas)})")
    statistics = _calc_statistics(datas, workers=workers)

    cache_dir.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(statistics.to_dict(), ensure_ascii=False))
    info_path.write_text(json.dumps(info, ensure_ascii=False))

    return statistics
