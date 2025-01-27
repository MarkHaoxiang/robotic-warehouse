from __future__ import annotations

from typing import TypeVar, Callable


In = TypeVar("In")
Out = TypeVar("Out")


def map_none(obj: In | None, cont: Callable[[In], Out]):
    return None if obj is None else cont(obj)
