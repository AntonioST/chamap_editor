from __future__ import annotations

from typing import Any

import numpy as np

from neurocarto.probe import ProbeDesp, ElectrodeDesp, M
from neurocarto.typing import *
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ControllerView

if SPHINX_BUILD:
    ProbeView = 'neurocarto.views.probe.ProbeView'

__all__ = [
    'new_channelmap',
    'capture_electrode',
    'captured_electrodes',
    'set_state_for_captured',
    'set_category_for_captured',
    'refresh_selection',
    'select_electrodes',
]


@doc_link()
def new_channelmap(controller: ControllerView, code: int | str) -> Any:
    """{ProbeView#reset()}"""
    app = controller.get_app()
    app.on_new(code)
    return app.probe_view.channelmap


@doc_link()
def capture_electrode(self: BlueprintFunctions, controller: ControllerView,
                      index: Array[int, N] | Array[bool, A],
                      state: list[int] = None):
    """{ProbeView#set_captured_electrodes()}"""
    electrodes = self.electrodes
    captured = [electrodes[int(it)] for it in np.arange(len(self.s))[index]]

    view = controller.get_app().probe_view
    if state is None:
        view.set_captured_electrodes(captured)
    else:
        for s in state:
            try:
                data = view.data_electrodes[s]
            except KeyError:
                pass
            else:
                view.set_captured_electrodes(captured, data)


@doc_link()
def captured_electrodes(controller: ControllerView, all=False) -> Array[int, N]:
    """{ProbeView#get_captured_electrodes_index()}"""
    view = controller.get_app().probe_view
    if all:
        captured = view.get_captured_electrodes_index(None, reset=False)
    else:
        captured = view.get_captured_electrodes_index(view.data_electrodes[ProbeDesp.STATE_USED], reset=False)

    return np.unique(captured)


@doc_link()
def set_state_for_captured(self: BlueprintFunctions, controller: ControllerView,
                           state: int,
                           index: Array[int, N] | Array[bool, A] = None):
    """{ProbeView#set_state_for_captured()}"""
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_state_for_captured(state)


@doc_link()
def set_category_for_captured(self: BlueprintFunctions, controller: ControllerView,
                              category: int,
                              index: Array[int, N] | Array[bool, A] = None):
    """{ProbeView#set_category_for_captured()}"""
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_category_for_captured(category)


@doc_link()
def refresh_selection(self: BlueprintFunctions, controller: ControllerView, selector: str = None):
    """{ProbeView#refresh_selection()}"""
    view = controller.get_app().probe_view

    old_select_args = dict(view.selecting_parameters)
    if selector is not None:
        view.selecting_parameters['selector'] = selector

    self.apply_blueprint(view.electrodes)

    try:
        view.refresh_selection()
        view.update_electrode()
        self.set_channelmap(view.channelmap)
    finally:
        view.selecting_parameters = old_select_args


@doc_link()
def select_electrodes(self: BlueprintFunctions, chmap: M = None,
                      blueprint: list[ElectrodeDesp] | Array[int, A] = None, **kwargs) -> M:
    """
    Run electrode selection for a channelmap based on the blueprint.

    :param self:
    :param chmap:
    :param blueprint:
    :param kwargs: selector extra parameters
    :return: selection result
    :see: {ProbeDesp#select_electrodes}
    """
    desp = self.probe

    if chmap is None:
        chmap = self.channelmap

    if blueprint is None:
        blueprint = self.blueprint()

    if isinstance(blueprint, np.ndarray):
        blueprint = self.apply_blueprint(blueprint=blueprint)

    return desp.select_electrodes(chmap, blueprint, **kwargs)
