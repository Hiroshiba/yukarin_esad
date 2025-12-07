"""型ユーティリティ"""

from pathlib import Path
from typing import Annotated

import fsspec
from fsspec.implementations.local import LocalFileSystem
from pydantic import BeforeValidator, PlainSerializer
from upath import UPath


def _to_upath(v: str):
    return UPath(v)


def _ser_upath(v: UPath | None):
    return None if v is None else str(v)


UPathField = Annotated[
    UPath,
    BeforeValidator(_to_upath),
    PlainSerializer(_ser_upath, return_type=str),
]


def to_local_path(p: UPath) -> Path:
    """リモートならキャッシュを作ってそのパスを、ローカルならそのままそのパスを返す"""
    if isinstance(p.fs, LocalFileSystem):
        return Path(p)
    obj = fsspec.open_local(
        "simplecache::" + str(p), simplecache={"cache_storage": "./hiho_cache/"}
    )
    if isinstance(obj, list):
        raise ValueError(f"複数のローカルパスが返されました: {p} -> {obj}")
    return Path(obj)
