from typing import Sequence

import numpy as np

from neurocarto.probe import M, E
from neurocarto.typing import *
from neurocarto.util.util_blueprint import BlueprintFunctions

__all__ = ['category_mask', 'invalid', 'merge_blueprint']


def category_mask(self: BlueprintFunctions,
                  blueprint: Array[int, A],
                  categories: int | list[int] = None) -> Array[bool, A]:
    if categories is None:
        categories = list(self.categories.values())
        try:
            categories.remove(self.CATE_UNSET)
        except ValueError:
            pass
        try:
            categories.remove(self.CATE_EXCLUDED)
        except ValueError:
            pass

    if isinstance(categories, (int, np.integer)):
        ret = blueprint == categories
    else:
        ret = np.zeros_like(blueprint, dtype=bool)
        for category in categories:
            np.logical_or(ret, blueprint == category, out=ret)

    return ret


def invalid(self: BlueprintFunctions,
            blueprint: Array[int, A],
            electrodes: Array[int, N] | Array[bool, A],
            value: int = None, *,
            overwrite: bool = False) -> Array[bool, A] | Array[int, A]:
    if electrodes.dtype == np.bool_:
        blueprint[electrodes]  # check shape
        protected = electrodes
    else:
        protected = np.zeros_like(blueprint, dtype=bool)
        protected[electrodes] = True

    all_electrodes = self.electrodes
    electrodes = [all_electrodes[it] for it in np.nonzero(protected)[0]]

    invalid_electrodes = self.index_blueprint(self.probe.invalid_electrodes(self.channelmap, electrodes, all_electrodes))
    invalid_mask = np.zeros_like(protected, dtype=bool)
    invalid_mask[invalid_electrodes] = True

    if value is None:
        if not overwrite:
            invalid_mask[protected] = False
        return invalid_mask
    else:
        ret = blueprint.copy()

        if overwrite:
            ret[invalid_mask] = value
        else:
            ret[invalid_mask & ~protected] = value

        return ret


def apply_electrode_mask(self: BlueprintFunctions,
                         masking: Array[bool, A],
                         electrodes: int | Sequence[E] | Array[bool, A] | Array[int, N] | M = None) -> Array[bool, A]:
    if electrodes is None:
        pass
    elif isinstance(electrodes, (int, np.integer)):
        keep = masking[electrodes]
        masking[:] = False
        masking[electrodes] = keep
    else:
        if isinstance(electrodes, type(self.channelmap)):
            electrodes = self.selected_electrodes(electrodes)

        if isinstance(electrodes, list):
            self.index_blueprint(electrodes)

        if not isinstance(electrodes, np.ndarray):
            raise TypeError()
        elif electrodes.dtype == np.bool_:
            masking = masking & electrodes
        else:
            keep = masking[electrodes]
            masking[:] = False
            masking[electrodes] = keep

    return masking


def merge_blueprint(self: BlueprintFunctions,
                    blueprint: Array[int, A],
                    other: Array[int, A] | BlueprintFunctions) -> Array[int, A]:
    if isinstance(other, BlueprintFunctions):
        if (other := other.blueprint()) is None:
            raise TypeError()

    n = len(self.s)
    if len(other) != n:
        raise ValueError()

    if len(blueprint) != n:
        raise ValueError()

    return np.where(blueprint != self.CATE_UNSET, blueprint, other)
