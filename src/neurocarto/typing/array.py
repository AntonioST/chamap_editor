from __future__ import annotations

from typing import Generic, TypeVar, Annotated, NewType

import numpy as np
from typing_extensions import Unpack, TypeVarTuple

__all__ = ['Array', 'DTYPE', 'SHAPE', 'A', 'N']

DTYPE = TypeVar('DTYPE')
SHAPE = TypeVarTuple('SHAPE')


class Array(Generic[DTYPE, Unpack[SHAPE]], np.ndarray):
    def __len__(self) -> int:
        raise NotImplementedError


"""
Python 3.12

class _Array[DTYPE, *SHAPE]():
    pass

type Array[DTYPE, *SHAPE] = _Array[DTYPE, *SHAPE] | np.ndarray

TODO not sure whether type checker happy or not.
"""

# common variables

A = Annotated[NewType('A', int), 'All electrode']
N = Annotated[NewType('N', int), 'Any number']
