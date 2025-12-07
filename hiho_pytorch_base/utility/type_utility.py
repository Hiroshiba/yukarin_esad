"""型ユーティリティ"""

from typing import Annotated

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
