"""
**Symbols used in this module**

* ``A`` All electrode
* ``N`` Any number
* ``DTYPE`` Any data type
* ``BLUEPRINT`` alias for ``Array[category:int, A]``
* ``Array[int, N]`` usually means an index array.
* ``Array[bool, A]`` usually means a blueprint mask.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, overload, Generic, Final, Any

import numpy as np
from typing_extensions import Self

from neurocarto.probe import ProbeDesp, M, E
from neurocarto.typing import *
from neurocarto.util.edit.checking import use_probe
from neurocarto.util.utils import doc_link, SPHINX_BUILD
from neurocarto.views.base import ControllerView, V

__all__ = ['BlueprintFunctions', 'ClusteringEdges', 'blueprint_function', 'use_probe']

BLUEPRINT = Array[int, N]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from neurocarto.util.edit.clustering import ClusteringEdges
    from neurocarto.util.probe_coor import ProbeCoordinate
    from neurocarto.views.atlas import Label

    ELECTRODES = Sequence[E]

elif SPHINX_BUILD:
    ProbeView = 'neurocarto.views.probe.ProbeView'
    NpxProbeDesp = 'neurocarto.probe_npx.desp.NpxProbeDesp'
    AtlasBrainView = 'neurocarto.views.atlas.AtlasBrainView'
    BlueprintScriptView = 'neurocarto.views.blueprint_script.BlueprintScriptView'
    ProbePlotElectrodeProtocol = 'neurocarto.views.blueprint_script.ProbePlotElectrodeProtocol'

    ELECTRODES = list[E]


def maybe_blueprint(self: BlueprintFunctions, a):
    """
    Is *a* a blueprint?

    :param self:
    :param a:
    :return: True if it may be a blueprint.
    """
    n = len(self.s)
    return isinstance(a, np.ndarray) and a.shape == (n,) and np.issubdtype(a.dtype, np.integer)


@doc_link(BlueprintFunctions='neurocarto.util.util_blueprint.BlueprintFunctions')
def blueprint_function(func=None, *, set_return=True):
    """
    Decorate a blueprint function to make it is able to direct apply function on
    the internal blueprint array.

    The function should have a signature ``(blueprint, ...) -> blueprint|None``.

    If the first parameter blueprint is given, it works as usually. ::

        bp.func(blueprint, ...)

    If the first parameter *blueprint* is omitted,
    use {BlueprintFunctions#blueprint()} as first arguments,
    and use {BlueprintFunctions#set_blueprint()} after it returns. ::

        bp.set_blueprint(bp.func(bp.blueprint(), ...))

    :param func:
    :param set_return: check the return and set the blueprint
    """

    def _decorator(func):
        @functools.wraps(func)
        def _blueprint_function(self: BlueprintFunctions, *args, **kwargs):
            if len(args) and maybe_blueprint(self, args[0]):
                return func(self, *args, **kwargs)
            else:
                blueprint = self.blueprint()
                ret = func(self, blueprint, *args, **kwargs)
                if set_return and maybe_blueprint(self, ret):
                    self.set_blueprint(ret)
                return ret

        return _blueprint_function

    if func is None:
        return _decorator
    else:
        return _decorator(func)


# noinspection PyMethodMayBeStatic
@doc_link()
class BlueprintFunctions(Generic[M, E]):
    """
    Provide blueprint manipulating functions. It is used by {BlueprintScriptView}.
    However, it also can be used independently, but without some view interacting functions.

    **channelmap functions**

    .. hlist::
        :columns: 2

        * {#check_probe()}
        * {#new_channelmap()}
        * {#add_electrodes()}
        * {#del_electrodes()}
        * {#selected_electrodes()}
        * {#set_channelmap()}
        * {#select_electrodes()}

    **blueprint functions**

    .. hlist::
        :columns: 2

        * {#blueprint()}
        * {#new_blueprint()}
        * {#blueprint_changed}
        * {#set_blueprint()}
        * {#apply_blueprint()}
        * {#from_blueprint()}
        * {#index_blueprint()}
        * {#load_blueprint()}
        * {#save_blueprint()}
        * {#set()}
        * {#unset()}
        * {#__setitem__()}
        * {#__delitem__()}
        * {#merge()}
        * {#count_categories()}
        * {#mask()}
        * {#invalid()}
        * {#move()}
        * {#move_i()}
        * {#find_clustering()}
        * {#clustering_edges()}
        * {#edge_rastering()}
        * {#fill()}
        * {#extend()}
        * {#reduce()}

    **electrode data processing**

    .. hlist::
        :columns: 2

        * {#load_data()}
        * {#interpolate_nan()}
        * {#draw()}

    **Probe view functions**

    Call methods from {ProbeView}.

    .. hlist::
        :columns: 2

        * {#capture_electrode()}
        * {#captured_electrodes()}
        * {#set_state_for_captured()}
        * {#set_category_for_captured()}
        * {#refresh_selection()}

    **Atlas Brain image view functions**

    Call methods from {AtlasBrainView}.

    .. hlist::
        :columns: 2

        * {#atlas_set_slice()}
        * {#atlas_add_label()}
        * {#atlas_add_label()}
        * {#atlas_del_label()}
        * {#atlas_clear_labels()}
        * {#atlas_set_transform()}
        * {#atlas_set_anchor()}
        * {#atlas_new_probe()}
        * {#atlas_set_anchor_on_probe()}

    **Blueprint script view functions**

    Call methods from {BlueprintScriptView}.

    .. hlist::
        :columns: 2

        * {#has_script()}
        * {#call_script()}
        * {#interrupt_script()}

    **UI communicating functions**

    .. hlist::
        :columns: 2

        * {#set_status_line()}
        * {#log_message()}
        * {#use_view()}

    **Miscellaneous**

    .. hlist::
        :columns: 2

        * {#clone()}
        * {#plot_channelmap()}
        * {#plot_blueprint()}
        * {#misc_profile_script()}

    **Category attributes**

    {BlueprintFunctions} supports 'CATE_*' electrode category attributes, which
    come from {#probe} without difference.
    Although {BlueprintFunctions} has declared some 'CATE_*' attributes,
    they do not have value. The actual value is delegated to the {#probe} via ``__getattr__``.

    """

    CATE_UNSET: int
    """electrode initial category"""

    CATE_SET: int
    """electrode pre-select category. Electrode must be selected"""

    CATE_EXCLUDED: int
    """electrode excluded category. Electrode must not be selected"""

    CATE_LOW: int
    """electrode low-priority category."""

    @doc_link()
    def __init__(self, probe: ProbeDesp[M, E], chmap: int | str | M | None):
        """
        initialization.

        Case when *chmap* is ``None``. In this case, {BlueprintFunctions} still be allowed to initialize.
        Although most of the functions will raise an error,
        it is okay because the caller (like {BlueprintScriptView}) would handle it,
        and it only happened when user forget to create a probe before running a script.

        :param probe:
        :param chmap: channelmap instance.
        """
        self.probe: Final[ProbeDesp[M, E]] = probe
        """probe"""

        if isinstance(chmap, (int, str)):
            chmap = probe.new_channelmap(chmap)
        self.channelmap: Final[M | None] = chmap
        """channelmap instance"""

        self.categories: Final[dict[str, int]] = probe.all_possible_categories()
        """categories mapping."""

        if chmap is not None:
            electrodes = probe.all_electrodes(chmap)
        else:
            electrodes = []

        self.electrodes: Final[list[E]] = electrodes
        """all available electrodes"""

        if chmap is not None:
            self.s: Final[Array[int, A]] = np.array([it.s for it in self.electrodes])
            self.x: Final[Array[int, A]] = np.array([it.x for it in self.electrodes])
            self.y: Final[Array[int, A]] = np.array([it.y for it in self.electrodes])
            self.dx: Final[float] = float(np.min(np.diff(np.unique(self.x))))
            self.dy: Final[float] = float(np.min(np.diff(np.unique(self.y))))
            if self.dx <= 0 or self.dy <= 0:
                raise ValueError(f'dx={self.dx}, dy={self.dy}')

            self._position_index: dict[tuple[int, int, int], int] = {
                (int(self.s[i]), int(self.x[i] / self.dx), int(self.y[i] / self.dy)): i
                for i in range(len(self.s))
            }

            self._blueprint: BLUEPRINT | None = np.array([it.category for it in electrodes])
        else:
            self._blueprint = None

        self._controller: ControllerView | None = None
        self._blueprint_changed = False

    def __getattr__(self, item: str):
        if item.startswith('CATE_'):
            if (ret := self.categories.get(item[5:], None)) is not None:
                return ret

        raise AttributeError(item)

    # noinspection PyFinal
    @doc_link()
    def clone(self, pure=False) -> Self:
        """
        Clone itself.

        **Pure**

        Whether the clone does not contain the controller to support UI functions,
        includes {ProbeView}, {AtlasBrainView} and {BlueprintScriptView} supporting.

        A pure clone has limited functions, but it is good for being pickled, and
        it could be passed in multiple processor for parallel computing.

        :param pure: Does not support UI function?
        :return: itself
        :raise RuntimeError: when probe is missing.
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        ret = object.__new__(BlueprintFunctions)
        ret.probe = self.probe  # type: ignore[misc]
        ret.channelmap = channelmap  # type: ignore[misc]
        ret.electrodes = self.electrodes  # type: ignore[misc]
        ret.categories = self.categories  # type: ignore[misc]

        ret.s = self.s  # type: ignore[misc]
        ret.x = self.x  # type: ignore[misc]
        ret.y = self.y  # type: ignore[misc]
        ret.dx = self.dx  # type: ignore[misc]
        ret.dy = self.dy  # type: ignore[misc]
        ret._position_index = self._position_index
        ret._blueprint = self._blueprint.copy()

        if pure:
            ret._controller = None
        else:
            ret._controller = self._controller

        return ret  # type: ignore[return-value]

    # ==================== #
    # channelmap functions #
    # ==================== #

    @doc_link()
    def add_electrodes(self, e: int | list[int] | Array[int, N] | Array[bool, A], *, overwrite=True):
        """
        Add electrode(s) *e* into the current channelmap.

        :param e: electrode index, index list, index array or index mask.
        :param overwrite: overwrite previous selected electrode.
        :raise RuntimeError: when probe is missing.
        :see: {ProbeDesp#add_electrode()}
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        electrodes = self.electrodes
        if isinstance(e, (int, np.integer)):
            es = [electrodes[int(e)]]
        elif isinstance(e, (list, tuple)):
            es = [electrodes[int(it)] for it in e]
        else:
            es = [electrodes[int(it)] for it in np.arange((len(electrodes)))[e]]

        for t in es:
            self.probe.add_electrode(channelmap, t, overwrite=overwrite)

    @doc_link()
    def del_electrodes(self, e: int | list[int] | Array[int, N] | Array[bool, A]):
        """
        delete electrode(s) *e* from the current channelmap.

        :param e: electrode index, index list, index array or index mask.
        :raise RuntimeError: when probe is missing.
        :see: {ProbeDesp#del_electrode()}
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        electrodes = self.electrodes
        if isinstance(e, (int, np.integer)):
            es = [electrodes[int(e)]]
        elif isinstance(e, (list, tuple)):
            es = [electrodes[int(it)] for it in e]
        else:
            es = [electrodes[int(it)] for it in np.arange((len(electrodes)))[e]]

        for t in es:
            self.probe.del_electrode(channelmap, t)

    @doc_link()
    def selected_electrodes(self, chmap=None) -> Array[int, N]:
        """
        The selected electrodes in the current channelmap.

        :return: electrode index array
        :see: {ProbeDesp#all_channels()}
        """
        if chmap is None:
            chmap = self.channelmap
        if chmap is None:
            return np.array([], dtype=int)
        return self.index_blueprint(self.probe.all_channels(chmap))

    @doc_link()
    def set_channelmap(self, chmap: M):
        """
        Apply the channelmap on the current channelmap.

        :param chmap:
        :raise RuntimeError: when probe is missing. call {#new_channelmap()} first.
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        # chmap may the same instance as self.channelmap
        # to prevent from we cannot get channels after clear_electrode()
        electrodes = self.probe.all_channels(chmap)
        self.probe.clear_electrode(channelmap)
        for t in electrodes:
            self.probe.add_electrode(channelmap, t)

    @doc_link()
    def select_electrodes(self, chmap=None, blueprint: ELECTRODES | BLUEPRINT = None, **kwargs) -> M:
        """
        Run electrode selection for a channelmap based on the blueprint.

        :param chmap:
        :param blueprint:
        :param kwargs: selector extra parameters
        :return: new channelmap
        :see: {ProbeDesp#select_electrodes}
        """
        from .edit.probe import select_electrodes
        return select_electrodes(self, chmap, blueprint, **kwargs)

    # =================== #
    # blueprint functions #
    # =================== #

    def blueprint(self) -> BLUEPRINT:
        """
        blueprint copy.

        :return:
        :raise RuntimeError: when probe is missing.
        """
        if self._blueprint is None:
            if self.channelmap is None:
                raise RuntimeError('probe missing')
            return np.full_like(self.s, self.CATE_UNSET, dtype=int)
        return self._blueprint.copy()

    def new_blueprint(self) -> BLUEPRINT:
        """
        new empty blueprint array.

        :return:
        :raise RuntimeError: when probe is missing.
        """
        if self.channelmap is None:
            raise RuntimeError('probe missing')

        return np.full_like(self.s, self.CATE_UNSET)

    @property
    def blueprint_changed(self) -> bool:
        """Has internal blueprint changed?"""
        return self._blueprint_changed

    def set_blueprint(self, blueprint: int | BLUEPRINT | ELECTRODES):
        """
        set blueprint.

        :param blueprint: a blueprint or a category value (reset all electrodes).
        :raise RuntimeError: when probe is missing.
        :raise ValueError: length mismatch
        """
        if self.channelmap is None:
            raise RuntimeError('probe missing')

        if isinstance(blueprint, int):
            if self._blueprint is None:
                self._blueprint = np.full_like(self.s, blueprint, dtype=int)
            else:
                self._blueprint[:] = blueprint

            self._blueprint_changed = True
            return

        if isinstance(blueprint, list):
            blueprint = self.from_blueprint(blueprint)

        if len(blueprint) != len(self.s):
            raise ValueError()

        self._blueprint = blueprint
        self._blueprint_changed = True

    @doc_link()
    def apply_blueprint(self, electrodes: ELECTRODES = None, blueprint: BLUEPRINT = None) -> ELECTRODES:
        """
        Apply blueprint back to electrode list.

        :param electrodes: electrode list
        :param blueprint:
        :return: *electrodes*
        :raise RuntimeError: when probe is missing.
        :see: {#from_blueprint()}
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        if blueprint is None:
            blueprint = self.blueprint()

        if electrodes is None:
            electrodes = self.electrodes
        else:
            for e in electrodes:
                e.state = ProbeDesp.STATE_UNUSED

        for e in self.probe.all_channels(channelmap, electrodes):
            for t in self.probe.invalid_electrodes(channelmap, e, electrodes):
                t.state = ProbeDesp.STATE_DISABLED
            e.state = ProbeDesp.STATE_USED

        c = {it.electrode: it for it in electrodes}
        for e, p in zip(self.electrodes, blueprint):
            if (t := c.get(e.electrode, None)) is not None:
                t.category = int(p)

        return electrodes

    @doc_link()
    def from_blueprint(self, electrodes: ELECTRODES) -> BLUEPRINT:
        """
        Get a blueprint from an electrode list.

        :param electrodes: electrode list
        :return:
        :raise RuntimeError: when probe is missing.
        :see: {#apply_blueprint()}
        """
        blueprint = self.new_blueprint()
        c = {it.electrode: it.category for it in electrodes}
        for i, e in enumerate(self.electrodes):
            if (category := c.get(e.electrode, None)) is not None:
                blueprint[i] = category
        return blueprint

    def index_blueprint(self, electrodes: ELECTRODES) -> Array[int, N]:
        """
        Get an electrode index array from an electrode list.

        :param electrodes:
        :return: electrode index array
        :raise RuntimeError: when probe is missing.
        """
        if self.channelmap is None:
            raise RuntimeError('probe missing')

        ret = []
        pos = self._position_index

        for e in electrodes:
            p = int(e.s), int(e.x / self.dx), int(e.y / self.dy)
            if (i := pos.get(p, None)) is not None:
                ret.append(i)

        return np.unique(np.array(ret, dtype=int))

    def load_blueprint(self, file: str | Path) -> BLUEPRINT:
        """
        Load the blueprint from the **file**.

        :param file: file.blueprint.npy
        :return:
        :raise RuntimeError: when probe is missing.
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        file = Path(file)
        if not file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        data = np.load(file)
        self.set_blueprint(self.probe.load_blueprint(data, channelmap))
        return self._blueprint

    def save_blueprint(self, file: str | Path, blueprint: BLUEPRINT = None):
        """
        Save blueprint to the **file**.

        :param file:
        :param blueprint:
        :raise RuntimeError: when probe is missing.
        """
        if (channelmap := self.channelmap) is None:
            raise RuntimeError('probe missing')

        if blueprint is None:
            blueprint = self._blueprint
            if blueprint is None:
                raise RuntimeError('nothing to save')
        else:
            if len(blueprint) != len(self.s):
                raise RuntimeError()

        file = Path(file)
        if not file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        s = self.probe.all_electrodes(channelmap)
        for e, c in zip(s, blueprint):
            e.category = c

        np.save(file, self.probe.save_blueprint(s))

    @blueprint_function
    @doc_link()
    def set(self, blueprint: BLUEPRINT, mask: int | list[int] | Array[bool, A] | Array[int, N], category: int) -> BLUEPRINT:
        """
        Set *category* on the blueprint with a *mask*.

        It is a {blueprint_function()} function.

        :param blueprint:
        :param mask: category value (or list), electrode mask or electrode index
        :param category:
        :return: a (copied) blueprint
        :raise RuntimeError: when probe is missing.
        """
        if self.channelmap is None:
            raise RuntimeError('probe missing')

        if len(blueprint) != len(self.s):
            raise ValueError()

        if isinstance(mask, (int, np.integer, list)):
            from .edit.category import category_mask as _mask
            mask = _mask(self, blueprint, mask)

        ret = blueprint.copy()
        ret[mask] = category
        return ret

    @blueprint_function
    @doc_link()
    def unset(self, blueprint: BLUEPRINT, mask: int | list[int] | Array[bool, A] | Array[int, N]) -> BLUEPRINT:
        """
        unset electrodes in the *blueprint* with a *mask*.

        It is a {blueprint_function()} function.

        :param blueprint:
        :param mask: category value, electrode mask or electrode index
        :return: a (copied) blueprint
        """
        return self.set(blueprint, mask, self.CATE_UNSET)

    @doc_link()
    def __setitem__(self, mask: int | Array[bool, A] | Array[int, N], category: int | str):
        """
        Set a *category* to the blueprint with a *mask*.
        The new *category* only apply on unset electrodes.
        If you want to overwrite the electrode's category, please use {#set()}.

        :param mask: category value, electrode mask or electrode index
        :param category:
        :see: {#merge()}, {#set()}
        """
        blueprint = self.blueprint()
        self.set_blueprint(self.merge(blueprint, self.set(blueprint, mask, category)))

    @doc_link()
    def __delitem__(self, mask: int | Array[bool, A] | Array[int, N]):
        """
        unset electrodes in the *blueprint* with a *mask*.

        :param mask: category value, electrode mask or electrode index
        :see: {#unset()}
        """
        self.set_blueprint(self.set(self.blueprint(), mask, self.CATE_UNSET))

    @overload
    def merge(self, blueprint: BLUEPRINT | BlueprintFunctions) -> BLUEPRINT:
        pass

    @overload
    def merge(self, blueprint: BLUEPRINT, other: BLUEPRINT | BlueprintFunctions) -> BLUEPRINT:
        pass

    def merge(self, blueprint, other=None) -> BLUEPRINT:
        """
        Merge two blueprints. The latter blueprint won't overwrite the former result.

        ``merge(blueprint)`` works like ``set_blueprint(ret := merge(blueprint(), blueprint)); ret``.

        :param blueprint: Array[category, N]
        :param other: blueprint Array[category, N]
        :return: blueprint Array[category, N]
        :raise RuntimeError: when only one argument is given and the probe is missing.
        :raise ValueError: incorrect length
        """
        from .edit.category import merge_blueprint
        set_return = False
        if other is None:
            blueprint, other = self._blueprint, blueprint
            if blueprint is None:
                blueprint = self.new_blueprint()
            set_return = True

        ret = merge_blueprint(self, blueprint, other)

        if set_return:
            self.set_blueprint(ret)
        return ret

    @blueprint_function(set_return=False)
    @doc_link()
    def count_categories(self, blueprint: BLUEPRINT, categories: int | list[int], mask: Array[bool, A] = None) -> int:
        """
        count number of electrode belonging to *categories*.

        It is a {blueprint_function()} function.

        :param blueprint:
        :param categories: category value or a category list.
        :param mask: a blueprint mask.
        :return: number of electrode in *categories* zone.
        """
        from .edit.category import category_mask as _mask
        t = _mask(self, blueprint, categories)
        if mask is not None:
            np.logical_and(t, mask, out=t)
        return np.count_nonzero(t)

    @blueprint_function(set_return=False)
    @doc_link()
    def mask(self, blueprint: BLUEPRINT, categories: int | list[int] = None) -> Array[bool, A]:
        """
        Masking electrode belong to the categories.

        It is a {blueprint_function()} function.

        :param blueprint:
        :param categories: If not given, use all categories except ``CATE_UNSET`` and ``CATE_EXCLUDED``.
        :return: a blueprint mask.
        """
        from .edit.category import category_mask
        return category_mask(self, blueprint, categories)

    @overload
    def invalid(self, blueprint: BLUEPRINT, *,
                electrodes: int | ELECTRODES | Array[bool, A] | Array[int, N] | M = None,
                categories: int | list[int] = None,
                overwrite: bool = False) -> Array[bool, A]:
        pass

    @overload
    def invalid(self, blueprint: BLUEPRINT, *,
                electrodes: int | ELECTRODES | Array[bool, A] | Array[int, N] | M = None,
                categories: int | list[int] = None,
                value: int,
                overwrite: bool = False) -> BLUEPRINT:
        pass

    @blueprint_function
    @doc_link()
    def invalid(self, blueprint: BLUEPRINT, *,
                electrodes: int | ELECTRODES | Array[bool, A] | Array[int, N] | M = None,
                categories: int | list[int] = None,
                value: int = None,
                overwrite: bool = False):
        """
        Masking or set value on invalid electrodes for electrode in categories.

        It is a {blueprint_function()} function.

        :param blueprint:
        :param electrodes: electrode index array, masking, or a channelmap (take selected electrodes).
        :param categories: only consider electrode categories in list.
        :param value: set value on invalid electrodes.
        :param overwrite: Does the electrode in categories are included in the mask.
        :return: a mask if *value* is omitted. Otherwise, a new blueprint.
        """
        from .edit.category import category_mask, invalid, apply_electrode_mask

        masking = category_mask(self, blueprint, categories)
        masking = apply_electrode_mask(self, masking, electrodes)

        return invalid(self, blueprint, masking, value, overwrite=overwrite)

    # ================== #
    # external functions #
    # ================== #

    def move(self, a: Array[DTYPE, N], *,
             tx: int = 0, ty: int = 0,
             mask: Array[bool, N] = None,
             axis: int = 0,
             init: float = 0) -> Array[DTYPE, N]:
        """
        Move blueprint zone.

        :param a: Array[V, ..., N, ...], where N means all electrodes
        :param tx: x movement in um.
        :param ty: y movement in um.
        :param mask: move electrode only in mask
        :param axis: index of N
        :param init: initial value V for *a*.
        :return: moved a (copied)
        """
        from .edit.moving import move
        return move(self, a, tx=tx, ty=ty, mask=mask, axis=axis, init=init)

    def move_i(self, a: Array[DTYPE, N], *,
               tx: int = 0, ty: int = 0,
               mask: Array[bool, N] = None,
               axis: int = 0,
               init: float = 0) -> Array[DTYPE, N]:
        """
        Move blueprint zone by steps of electrode interval space.

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx: number of dx
        :param ty: number of dy
        :param mask: move electrode only in mask
        :param axis: index of N
        :param init: initial value V for *a*.
        :return: moved a (copied)
        """
        from .edit.moving import move_i
        return move_i(self, a, tx=tx, ty=ty, mask=mask, axis=axis, init=init)

    def find_clustering(self, blueprint: BLUEPRINT,
                        categories: int | list[int] = None, *,
                        diagonal=True) -> BLUEPRINT:
        """
        find electrode clustering with the same category zone.

        :param blueprint: Array[category, N]
        :param categories: only for given categories.
        :param diagonal: does surrounding includes electrodes on diagonal?
        :return: Array[int, N]
        """
        from .edit.clustering import find_clustering
        return find_clustering(self, blueprint, categories, diagonal=diagonal)

    @doc_link()
    def clustering_edges(self, blueprint: BLUEPRINT,
                         categories: int | list[int] = None) -> list[ClusteringEdges]:
        """
        For each clustering, and calculate its edges.

        :param blueprint:
        :param categories:
        :return: list of edge
        """
        from .edit.clustering import clustering_edges
        return clustering_edges(self, blueprint, categories)

    def edge_rastering(self, edges: ClusteringEdges | list[ClusteringEdges], *,
                       fill=False, overwrite=False) -> BLUEPRINT:
        """
        For given edges, put them on the blueprint.

        :param edges:
        :param fill: fill the area.
        :param overwrite: latter result overwrite previous results
        :return: blueprint
        """
        from .edit.clustering import edge_rastering
        return edge_rastering(self, edges, fill=fill, overwrite=overwrite)

    @blueprint_function
    @doc_link()
    def fill(self, blueprint: BLUEPRINT,
             categories: int | list[int] = None, *,
             threshold: int = None,
             gap: int | None = 1,
             unset: bool = False) -> BLUEPRINT:
        """
        fill each category zone as a rectangle zone.

        It is a {blueprint_function()} function.

        :param blueprint: Array[category, N]
        :param categories: only consider given categories.
        :param threshold: only consider area which size larger than threshold.
        :param gap: fill the gap below (abs(y) <= gap). Use ``None``, fill an zone as a prefect rectangle.
        :param unset: unset small zone (depends on threshold)
        :return: blueprint Array[category, N]
        """
        from .edit.moving import fill
        return fill(self, blueprint, categories, threshold=threshold, gap=gap, unset=unset)

    @blueprint_function
    @doc_link()
    def extend(self, blueprint: BLUEPRINT,
               category: int,
               step: int | tuple[int, int],
               value: int = None, *,
               threshold: int | tuple[int, int] = None,
               bi: bool = True,
               overwrite: bool = False) -> BLUEPRINT:
        """
        extend the zone of each category's zone.

        It is a {blueprint_function()} function.

        :param blueprint: Array[category, N]
        :param category: on which category zone.
        :param step: expend step on y or (x, y)
        :param value: a category value used in the extending.
        :param threshold: Positive value: extend the zone which size larger than threshold.
            Negative value: extend the zone which size smaller than threshold.
            A tuple: extend the zone which size in a range.
        :param bi: both position and negative steps direction
        :param overwrite: overwrite category value. By default, only change the unset electrode.
        :return: a modified blueprint copy.
        """
        from .edit.moving import extend
        return extend(self, blueprint, category, step, value, threshold=threshold, bi=bi, overwrite=overwrite)

    @blueprint_function
    @doc_link()
    def reduce(self, blueprint: BLUEPRINT,
               category: int,
               step: int | tuple[int, int], *,
               threshold: int | tuple[int, int] = None,
               bi: bool = True) -> BLUEPRINT:
        """
        reduce the size of each category's zone.

        It is a {blueprint_function()} function.

        :param blueprint: Array[category, N]
        :param category: on which category zone.
        :param step: reduce step on y or (x, y)
        :param threshold: Positive value: extend the zone which size larger than threshold.
            Negative value: extend the zone which size smaller than threshold.
            A tuple: extend the zone which size in a range.
        :param bi: both position and negative steps direction
        :return: a modified blueprint copy.
        """
        from .edit.moving import reduce
        return reduce(self, blueprint, category, step, threshold=threshold, bi=bi)

    # ==================== #
    # data process methods #
    # ==================== #

    @doc_link()
    def load_data(self, file: str | Path) -> Array[float, A]:
        """
        Load a numpy array that can be parsed by {ProbeDesp#load_blueprint()}.
        The data value is read from category value for electrodes.

        Because E's category is expected as an int, this view also take it as an int by default.

        For the Neuropixels, {NpxProbeDesp} use the numpy array in this form:

           Array[int, E, (shank, col, row, state, category)]

        :param file: data file
        :return: data array.
        """
        from .edit.data import load_data
        return load_data(self, file)

    def interpolate_nan(self, a: Array[float, N],
                        kernel: int | tuple[int, int] = 1,
                        f: str | Callable[[Array[float, N]], float] = 'mean') -> Array[float, N]:
        """
        Interpolate the NaN value in the data *a*.

        :param a: data array.
        :param kernel: kernel size.
        :param f: interpolate method. Default use mean.
        :return: interpolated data.
        """
        from .edit.data import interpolate_nan
        return interpolate_nan(self, a, kernel, f)

    # ==================== #
    # controller functions #
    # ==================== #

    @doc_link(
        get_probe_desp='neurocarto.probe.get_probe_desp',
        RequestChannelmapTypeError='neurocarto.util.edit.checking.RequestChannelmapTypeError',
    )
    def check_probe(self, probe: str | type[ProbeDesp] | None = None,
                    chmap_code: int = None, *, error=True):
        """
        Check current used probe is type of *probe*.

        If it is managed by {BlueprintScriptView}, {RequestChannelmapTypeError} could be captured,
        and the request channelmap ({#new_channelmap()}) will be created when needed.

        In another way, decorator {use_probe()} can be used to annotate the probe request on a script,
        then {BlueprintScriptView} can handle the probe creating and checking before running the script.

        :param probe: request probe. It could be family name (via {get_probe_desp()}), {ProbeDesp} type or class name.
            It ``None``, checking a probe has created, and its type doesn't matter.
        :param chmap_code: request channelmap code
        :param error: raise an error when the checking fail.
        :return: test success.
        :raise RequestChannelmapTypeError: when check failed.
        """
        from .edit.checking import check_probe, RequestChannelmapTypeError

        try:
            check_probe(self, probe, chmap_code)
        except RequestChannelmapTypeError:
            if error:
                raise
            return False
        else:
            return True

    @doc_link()
    def use_view(self, view: str | type[V]) -> V | None:
        """
        Get corresponding {ViewBase} instance if activated.

        Note:
            Avoiding import ``V`` at the global that might cause ``ImportError``,
            either using type name or using local import.

        :param view: view type or its type name.
        :return:
        """
        if (controller := self._controller) is not None:
            return controller.get_view(view)
        return None

    @doc_link(BlueprintScriptView='neurocarto.views.blueprint_script.BlueprintScriptView')
    def draw(self, a: Array[float, A] | None):
        """
        Send a drawable data array *a*  to a {BlueprintScriptView}.

        :param a: Array[float, E], where E is all electrodes
        """
        from .edit.actions import draw
        if (controller := self._controller) is not None:
            draw(self, controller, a)

    @doc_link()
    def set_status_line(self, message: str, *, decay: float = None):
        """

        :param message: message
        :param decay: after give seconds, clear the message.
        :see: {ViewBase#set_status()}
        """
        from .edit.actions import set_status_line
        if (controller := self._controller) is not None:
            set_status_line(controller, message, decay=decay)

    @doc_link(CartoApp='neurocarto.main_app.CartoApp')
    def log_message(self, *message: str):
        """
        Send messages to log area in GUI.

        :param message:
        :see: {CartoApp#log_message()}
        """
        from .edit.actions import log_message
        if (controller := self._controller) is not None:
            log_message(controller, *message)

    @doc_link()
    def has_script(self, script: str) -> bool:
        """
        Check whether *script* is existed in {BlueprintScriptView}'s action list.

        :param script: script name.
        :return:
        """
        from .edit.actions import has_script
        if (controller := self._controller) is not None:
            return has_script(controller, script)
        return False

    @doc_link(
        use_probe='neurocarto.util.edit.checking.use_probe',
        RequestChannelmapTypeError='neurocarto.util.edit.checking.RequestChannelmapTypeError',
    )
    def call_script(self, script: str, /, *args, **kwargs):
        """
        run script in {BlueprintScriptView}.

        There are some difference behavior when running a script
        from this method and {BlueprintScriptView#run_script}.

        * Both check script are up-to-date.
        * This method does not handle {use_probe()} and {RequestChannelmapTypeError}
        * This method does not print same logging message as {RequestChannelmapTypeError}.
        * This method does not record history step.
        * Both are allow a generator from the script, but

          * this method pass the generator to the {BlueprintScriptView}.
          * {BlueprintScriptView} mark the target script as interruptable, instead of this method.
          * this method will lost control on the generator.

        :param script: script name in {BlueprintScriptView} action list.
        :param args: script positional arguments.
        :param kwargs: script keyword arguments.
        """
        from .edit.actions import call_script
        if (controller := self._controller) is not None:
            call_script(self, controller, script, *args, **kwargs)

    @doc_link()
    def interrupt_script(self, script: str) -> bool:
        """
        Interrupt script.

        :param script: script name.
        :return: is interrupt success?
        :see: {BlueprintScriptView#interrupt_script()}
        """
        from .edit.actions import interrupt_script
        if (controller := self._controller) is not None:
            return interrupt_script(controller, script)
        else:
            return False

    def set_script_input(self, script: str | None, *text: str | None):
        """
        Set the script input for *script*.

        :param script: script name. If ``None``, use current script.
        :param text: each argument
        """
        from .edit.actions import set_script_input
        if (controller := self._controller) is not None:
            set_script_input(controller, script, *text)

    # ================= #
    # ProbeView related #
    # ================= #

    @doc_link()
    def new_channelmap(self, code: int | str) -> M:
        """
        Create a new channelmap with type *code*.

        :param code: channelmap type code.
        :return: new channelmap instance.
        :see: {ProbeView#reset()}
        """
        from .edit.probe import new_channelmap
        if (controller := self._controller) is not None:
            return new_channelmap(controller, code)
        else:
            raise RuntimeError()

    @doc_link()
    def capture_electrode(self, index: Array[int, N] | Array[bool, A],
                          state: list[int] = None):
        """
        Capture electrodes.

        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :param state: restrict electrodes on given states.
        :see: {ProbeView#set_captured_electrodes()}
        """
        from .edit.probe import capture_electrode
        if (controller := self._controller) is not None:
            capture_electrode(self, controller, index, state)

    @doc_link()
    def captured_electrodes(self, all=False) -> Array[int, N]:
        """
        get captured electrodes.

        :param all: capture excluded electrodes?
        :return: index of captured electrodes.
        :see: {ProbeView#get_captured_electrodes_index()}
        """
        from .edit.probe import captured_electrodes
        if (controller := self._controller) is not None:
            return captured_electrodes(controller, all)
        else:
            return np.array([], dtype=int)

    @doc_link()
    def set_state_for_captured(self, state: int,
                               index: Array[int, N] | Array[bool, A] = None):
        """
        Set state for captured electrodes.

        :param state: a state value
        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :see: {#capture_electrode()}
        :see: {ProbeView#set_state_for_captured()}
        """
        from .edit.probe import set_state_for_captured
        if (controller := self._controller) is not None:
            set_state_for_captured(self, controller, state, index)

    @doc_link()
    def set_category_for_captured(self, category: int,
                                  index: Array[int, N] | Array[bool, A] = None):
        """
        Set category for captured electrodes.

        :param category: a category value
        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :see: {#capture_electrode()}
        :see: {ProbeView#set_category_for_captured()}
        """
        from .edit.probe import set_category_for_captured
        if (controller := self._controller) is not None:
            set_category_for_captured(self, controller, category, index)

    @doc_link()
    def refresh_selection(self, selector: str = None):
        """
        refresh electrode selection base on current blueprint.

        :param selector:
        :see: {ProbeView#refresh_selection()}
        """
        from .edit.probe import refresh_selection
        if (controller := self._controller) is not None:
            refresh_selection(self, controller, selector)

    # ====================== #
    # AtlasBrainView related #
    # ====================== #

    @doc_link()
    def atlas_get_slice(self, *, um=False) -> tuple[str | None, int | None]:
        """
        Get atlas brain image projection view and slicing plane.

        :param um: is plane index in return um? If so, then use bregma as origin.
        :return: tuple of (projection name, plane index)
        """
        from .edit.atlas import atlas_get_slice
        if (controller := self._controller) is not None:
            return atlas_get_slice(controller, um=um)
        return None, None

    @doc_link()
    def atlas_set_slice(self, view: str = None, plane: int = None, *, um=False):
        """
        Set atlas image projection view and slicing plane.

        :param view: 'coronal', 'sagittal', or 'transverse'
        :param plane: plane index
        :param um: is *plane* um? If so, then use bregma as origin.
        """
        from .edit.atlas import atlas_set_slice
        if (controller := self._controller) is not None:
            atlas_set_slice(controller, view, plane, um=um)

    @doc_link()
    def atlas_add_label(self, text: str, pos: tuple[float, float] | tuple[float, float, float] = None, *,
                        origin: str = 'bregma', color: str = 'cyan', replace=True) -> Label | None:
        """
        Add a label on atlas image view.

        :param text: text content.
        :param pos: text position
        :param origin: origin reference point
        :param color: label color
        :param replace: replace label which has same text content
        :return: label
        :see: {AtlasBrainView#add_label()}
        """
        from .edit.atlas import atlas_add_label
        if (controller := self._controller) is not None:
            return atlas_add_label(controller, text, pos, origin=origin, color=color, replace=replace)
        return None

    @doc_link()
    def atlas_get_label(self, index: int | str) -> Label | None:
        """
        Get the label from atlas image view.

        :param index: label index or its content
        :see: {AtlasBrainView#get_label()}
        """
        from .edit.atlas import atlas_get_label
        if (controller := self._controller) is not None:
            return atlas_get_label(controller, index)
        return None

    @doc_link()
    def atlas_focus_label(self, label: int | str | Label):
        """
        Move slice to the label's position.

        Note: Only works on the labels which its origin is referring on the bregma.
        Otherwise, nothing will happen.

        :param label: label index, content or a {Label}.
        :see: {AtlasBrainView#focus_label()}
        """
        from .edit.atlas import atlas_focus_label
        if (controller := self._controller) is not None:
            atlas_focus_label(controller, label)

    @doc_link()
    def atlas_del_label(self, i: int | str | Label | list[int | str | Label]):
        """
        Remove labels from atlas image.

        :param i: index, or label text or list of them.
        :see: {AtlasBrainView#del_label()}
        """
        from .edit.atlas import atlas_del_label
        if (controller := self._controller) is not None:
            atlas_del_label(controller, i)

    @doc_link()
    def atlas_clear_labels(self):
        """
        Clear all labels on the atlas image.

        :see: {AtlasBrainView#clear_labels()}
        """
        from .edit.atlas import atlas_clear_labels
        if (controller := self._controller) is not None:
            atlas_clear_labels(controller)

    # ================================= #
    # AtlasBrainView coordinate related #
    # ================================= #

    @doc_link()
    def atlas_set_transform(self, p: tuple[float, float] = None,
                            s: float | tuple[float, float] = None,
                            rt: float = None):
        """
        updating atlas image transforming.

        :param p: center position (x, y)
        :param s: scaling (sx, sy)
        :param rt: rotating degree
        :see: {BoundView#update_boundary_transform()}
        """
        from .edit.atlas import atlas_set_transform
        if (controller := self._controller) is not None:
            atlas_set_transform(controller, p, s, rt)

    @doc_link()
    def atlas_set_anchor(self, p: tuple[float, float], a: tuple[float, float] = (0, 0)):
        """
        Update atlas image boundary transform to move *a* onto *p*.

        :param p: target point on figure. figure (probe) origin as origin.
        :param a: anchor point on image, center point as origin.
        """
        from .edit.atlas import atlas_set_anchor
        if (controller := self._controller) is not None:
            atlas_set_anchor(controller, p, a)

    @doc_link()
    def atlas_new_probe(self,
                        ap: float, dv: float, ml: float,
                        shank: int = 0,
                        rx: float = 0, ry: float = 0, rz: float = 0,
                        depth: float = 0,
                        ref: str = 'bregma') -> ProbeCoordinate | None:
        """
        Create a probe coordinate instance.

        :param ap: ap um, from ref (default bregma).
        :param dv: dv um, from ref (default bregma).
        :param ml: ml um, from ref (default bregma).
        :param shank: shank index
        :param rx: ap-axis rotate
        :param ry: dv-axis rotate
        :param rz: ml-axis rotate
        :param depth: insert depth
        :param ref: reference origin.
        :return: a probe coordinate. ``None`` if origin not set.
        """
        from .edit.atlas import atlas_new_probe
        if (controller := self._controller) is not None:
            return atlas_new_probe(controller, ap, dv, ml, shank, rx, ry, rz, depth, ref)
        return None

    @doc_link()
    def atlas_current_probe(self, shank: int = 0, ref: str = 'bregma') -> ProbeCoordinate | None:
        """
        Get the current coordinate of the probe.

        The dv value of the returned coordinate always zero.

        :param shank: which shank is the returned coordinate is based on.
        :param ref: which origin is the returned coordinate is referring to.
        :return: a probe coordinate. ``None`` if origin not found.
        """
        from .edit.atlas import atlas_current_probe
        if (controller := self._controller) is not None:
            return atlas_current_probe(self, controller, shank, ref)
        return None

    @doc_link()
    def atlas_set_anchor_on_probe(self, coor: ProbeCoordinate):
        """
        Update atlas image boundary transform to anchor insertion point onto the probe.

        :param coor: probe coordinate
        """
        from .edit.atlas import atlas_set_anchor_on_probe
        if (controller := self._controller) is not None:
            atlas_set_anchor_on_probe(self, controller, coor)

    # =================== #
    # matplotlib plotting #
    # =================== #

    @doc_link(PltImageView='neurocarto.views.image_plt.PltImageView')
    def plot_channelmap(self, channelmap: M = None,
                        color: Any = 'black', *,
                        ax: Axes = None, **kwargs):
        """
        call {ProbeDesp} primary plotting method for plot a channelmap
        with a matplotlib axes.

        If *ax* is ``None``, create one with the rc file used by {PltImageView}.

        :param channelmap: a channelmap instance
        :param color: color of selected electrodes, where color is used by matplotlib.
        :param ax: matplotlib.Axes
        :param kwargs:
        :raise TypeError: Probe not a {ProbePlotElectrodeProtocol}
        """
        from .edit.plot import plot_blueprint

        if channelmap is None:
            channelmap = self.channelmap

        blueprint = self.new_blueprint()
        blueprint[self.selected_electrodes(channelmap)] = self.CATE_SET
        plot_blueprint(self, blueprint, {self.CATE_SET: color}, ax=ax, **kwargs)

    @blueprint_function(set_return=False)
    @doc_link(PltImageView='neurocarto.views.image_plt.PltImageView')
    def plot_blueprint(self, blueprint: BLUEPRINT = None,
                       colors: dict[int, Any] = None, *,
                       ax: Axes = None, **kwargs):
        """
        call {ProbeDesp} primary plotting method for plotting a blueprint
        with a matplotlib axes.

        If *ax* is ``None``, create one with the rc file used by {PltImageView}.

        It is a {blueprint_function()} function.

        :param blueprint: Array[category:int, E], where E means all electrodes
        :param colors: categories color {category: color}, where color is used by matplotlib.
        :param ax: matplotlib.Axes
        :param kwargs:
        :raise TypeError: Probe not a {ProbePlotElectrodeProtocol}
        """
        from .edit.plot import plot_blueprint
        if blueprint is None:
            blueprint = self.blueprint()
        plot_blueprint(self, blueprint, colors, ax=ax, **kwargs)

    # ============= #
    # Miscellaneous #
    # ============= #

    @doc_link()
    def misc_profile_script(self, script: str, /, *args, **kwargs):
        """
        Call script under cProfile.

        This method works almost as same as {#call_script()}, but

        * under profiling
        * This method does handle generator, but ignoring the yield value. Just loop until it stops.
        * generate a profiling result named ``profile-SCRIPT.dat`` at the cache directory.

        The ``profile-SCRIPT.dat`` file can use following command to generate a figure:

        .. code-block:: bash

            python -m gprof2dot -f pstats profile-SCRIPT.dat | dot -T png -o profile-SCRIPT.png

        :param script: script name
        :param args: script positional arguments
        :param kwargs: script keyword arguments
        :see: {#call_script()}
        """
        from .edit.actions import profile_script
        if (controller := self._controller) is not None:
            profile_script(self, controller, script, *args, **kwargs)
        else:
            raise RuntimeError()
