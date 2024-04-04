from collections.abc import Callable
from pathlib import Path

import numpy as np

from neurocarto.typing import *
from neurocarto.util.util_blueprint import BlueprintFunctions
from .moving import move_i

__all__ = ['load_data', 'interpolate_nan']


def load_data(self: BlueprintFunctions, file: str | Path) -> Array[float, A]:
    e = self.probe.all_electrodes(self.channelmap)
    for t in e:
        t.category = np.nan
    e = self.probe.load_blueprint(file, e)
    return np.array([it.category for it in e], dtype=float)


def interpolate_nan(self: BlueprintFunctions,
                    a: Array[float, N],
                    kernel: int | tuple[int, int] = 1,
                    f: str | Callable[[Array[float, N]], float] = 'mean') -> Array[float, N]:
    if isinstance(f, str):
        if f == 'mean':
            f = np.nanmean
        elif f == 'median':
            f = np.nanmedian
        elif f == 'min':
            f = np.nanmin
        elif f == 'max':
            f = np.nanmax
        else:
            raise ValueError()

    if not np.any(m := np.isnan(a)):
        return a

    match kernel:
        case 0 | (0, 0):
            return a
        case int(y) if y > 0:
            kernel = (0, y)
        case (int(x), int(y)) if x >= 0 and y >= 0:
            pass
        case int() | (int(), int()):
            raise ValueError()
        case _:
            raise TypeError()

    r = []
    for tx in range(-kernel[0], kernel[0] + 1):
        for ty in range(-kernel[1], kernel[1] + 1):
            r.append(move_i(self, a, tx=tx, ty=ty, init=np.nan))

    r = f(r, axis=0)

    ret = a.copy()
    ret[m] = r[m]
    return ret
