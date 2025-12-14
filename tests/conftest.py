"""pytestの共通設定と自動テストデータ生成"""

import os
from pathlib import Path
from typing import Literal, cast

import pytest
from upath import UPath

from hiho_pytorch_base.config import Config
from tests.test_utils import setup_data_and_config


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """テスト環境のセットアップ"""
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(scope="session")
def input_data_dir() -> Path:
    """入力データディレクトリのパス"""
    return Path(__file__).parent / "input_data"


@pytest.fixture(scope="session")
def output_data_dir() -> UPath:
    """出力データディレクトリのパス"""
    return UPath(__file__).parent / "output_data"


@pytest.fixture(scope="session")
def base_config_path(input_data_dir: Path) -> Path:
    """ベース設定ファイルのパス"""
    return input_data_dir / "base_config.yaml"


@pytest.fixture(scope="session", autouse=True)
def data_and_config(base_config_path: Path, output_data_dir: UPath) -> Config:
    """データディレクトリと学習テスト用の設定のセットアップ"""
    data_dir = output_data_dir / "train_data"
    return setup_data_and_config(base_config_path, data_dir)


@pytest.fixture(params=["meanflow", "rectified_flow"])
def flow_type(
    request: pytest.FixtureRequest,
) -> Literal["meanflow", "rectified_flow"]:
    """Flowの種類"""
    return cast(Literal["meanflow", "rectified_flow"], request.param)


@pytest.fixture
def train_config(
    data_and_config: Config, flow_type: Literal["meanflow", "rectified_flow"]
) -> Config:
    """学習テスト用設定"""
    config = data_and_config.model_copy(deep=True)
    config.dataset.flow_type = flow_type
    config.network.flow_type = flow_type
    config.model.flow_type = flow_type
    config.validate_config()
    return config


@pytest.fixture(scope="session")
def train_output_dir(output_data_dir: UPath) -> UPath:
    """学習結果ディレクトリのパス"""
    output_dir = output_data_dir / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
