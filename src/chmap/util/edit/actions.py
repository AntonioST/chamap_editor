from typing import Any

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp, get_probe_desp
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.views.base import ViewBase, ControllerView
from chmap.views.data import DataHandler

__all__ = [
    'RequestChannelmapTypeError',
    'check_probe',
    'new_channelmap',
    'log_message',
    'draw'
]


class RequestChannelmapTypeError(RuntimeError):
    def __init__(self, probe: str | type[ProbeDesp], chmap_code: int | None):
        """

        :param probe: request probe type.
        :param chmap_code: request channelmap code
        """
        self.probe = probe
        self.chmap_code = chmap_code

        if isinstance(probe, str):
            message = f'Request Probe[{probe}]'
        else:
            message = f'Request {probe.__name__}'

        if chmap_code is not None:
            message += f'[{chmap_code}].'
        else:
            message += '.'

        super().__init__(message)

    @property
    def probe_name(self) -> str:
        if isinstance(self.probe, type):
            return self.probe.__name__

        elif isinstance(self.probe, str):
            return self.probe

        else:
            raise RuntimeError()

    def check_probe(self, probe: ProbeDesp) -> bool:
        if isinstance(self.probe, type):
            return isinstance(probe, self.probe)

        elif isinstance(probe, str):
            return type(probe).__name__ == self.probe

        else:
            raise RuntimeError()


def check_probe(self: BlueprintFunctions, probe: str | type[ProbeDesp], chmap_code: int = None):
    """
    check request probe type and channelmap code.

    :param self:
    :param probe: request probe type.
    :param chmap_code: request channelmap code
    :return: `None`
    :raise RequestChannelmapTypeError: when check failed.
    """
    current_probe = getattr(self, 'probe', None)
    current_chmap = getattr(self, 'chmap', None)

    if isinstance(probe, type):
        test = isinstance(current_probe, probe)

    elif isinstance(probe, str):
        test = type(current_probe).__name__ == probe
        if not test:
            try:
                probe_type = get_probe_desp(probe)
            except BaseException:
                pass
            else:
                probe = probe_type
                test = isinstance(current_probe, probe)
    else:
        raise TypeError()

    if test and chmap_code is not None:
        if (code := self.probe.channelmap_code(current_chmap)) is None:
            test = False
        else:
            test = chmap_code == code

    if not test:
        raise RequestChannelmapTypeError(probe, chmap_code)


def new_channelmap(controller: ControllerView, code: int | str) -> Any:
    app = controller.get_app()
    app.on_new(code)
    return app.probe_view.channelmap


def log_message(controller: ControllerView, *message: str):
    if isinstance(controller, ViewBase):
        controller.log_message(*message)


def draw(self: BlueprintFunctions, controller: ControllerView,
         a: NDArray[np.float_] | None, *,
         view: str | type[ViewBase] = None):
    if isinstance(controller, DataHandler):
        controller.on_data_update(self.probe, self.probe.all_electrodes(self.chmap), a)
    elif isinstance(view_target := controller.get_view(view), DataHandler):
        view_target.on_data_update(self.probe, self.probe.all_electrodes(self.chmap), a)
