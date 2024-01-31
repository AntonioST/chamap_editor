import logging
import time
from collections.abc import Iterable
from typing import Any

from bokeh.models import ColumnDataSource, GlyphRenderer, tools
from bokeh.plotting import figure as Figure

from chmap.probe import ProbeDesp, E, M
from chmap.util.bokeh_app import run_timeout
from chmap.util.bokeh_util import as_callback

__all__ = ['ProbeView']


class ProbeView:
    """
    Probe view.

    It is a special view component handled by ChannelMapEditorApp, so we do not inherit from ViewBase.

    """

    STYLES: dict[int | str, dict[str, Any]] = {
        ProbeDesp.STATE_USED: dict(color='green'),
        ProbeDesp.STATE_UNUSED: dict(color='black'),
        ProbeDesp.STATE_FORBIDDEN: dict(color='red', size=2, alpha=0.2),
        'highlight': dict(color='yellow', size=6, alpha=0.5)
    }

    data_electrodes: dict[int, ColumnDataSource]
    render_electrodes: dict[int, GlyphRenderer]

    data_highlight: ColumnDataSource
    render_highlight: GlyphRenderer

    def __init__(self, desp: ProbeDesp[M, E]):
        self.logger = logging.getLogger('chmap.view.probe')

        self.logger.debug('init(%s)', type(desp).__name__)
        self.probe: ProbeDesp[M, E] = desp
        self.channelmap: M | None = None
        self.electrodes: list[E] | None = None
        self._e2i: dict[E, int] = {}

        self.data_electrodes = {}
        for state in (ProbeDesp.STATE_UNUSED, ProbeDesp.STATE_USED, ProbeDesp.STATE_FORBIDDEN):
            self.data_electrodes[state] = ColumnDataSource(data=dict(x=[], y=[], e=[], c=[]))
            self.data_electrodes[state].selected.on_change('indices', as_callback(self.on_select, state=state))

        self.data_highlight = ColumnDataSource(data=dict(x=[], y=[], e=[]))
        self.selecting_parameters = {}

    def plot(self, f: Figure):
        self.logger.debug('setup(figure)')
        self.render_highlight = f.scatter(
            x='x', y='y', source=self.data_highlight, **self.STYLES.get('highlight', {})
        )

        self.render_electrodes = {
            state: f.scatter(
                x='x', y='y', source=data, **self.STYLES.get(state, {})
            )
            for state, data in self.data_electrodes.items()
        }

    def setup_tools(self) -> list[tools.Tool]:
        self.logger.debug('setup(tool)')

        return [
            tools.BoxSelectTool(
                description='select electrode',
                renderers=list(self.render_electrodes.values())
            ),
            tools.HoverTool(
                description="electrode information",
                renderers=[
                    self.render_electrodes[ProbeDesp.STATE_USED],
                    self.render_electrodes[ProbeDesp.STATE_UNUSED],
                    self.render_electrodes[ProbeDesp.STATE_FORBIDDEN],
                ],
                tooltips=[  # hover
                    ('Channel', "@c"),
                    ("(x,y)", "($x, $y)"),
                ]
            ),
        ]

    def channelmap_desp(self) -> str:
        return self.probe.channelmap_desp(self.channelmap)

    def reset(self, chmap: int | M = None):
        """
        Reset channelmap.

        :param chmap: channelmap code
        """
        if chmap is None:
            self.logger.debug('reset()')
            channelmap = self.probe.new_channelmap(self.channelmap)
        elif isinstance(chmap, int):
            self.logger.debug('reset(%d)', chmap)
            channelmap = self.probe.new_channelmap(chmap)
        else:
            self.logger.debug('reset(%s)', type(chmap).__name__)
            channelmap = self.probe.copy_channelmap(chmap)

        self.channelmap = channelmap
        self.electrodes = self.probe.all_electrodes(channelmap)
        self._reset_electrode_state()

        self._e2i = {}
        for i, e in enumerate(self.electrodes):  # type: int, E
            self._e2i[e] = i

    def _reset_electrode_state(self):
        for e in self.electrodes:
            e.state = ProbeDesp.STATE_UNUSED

        c = self.probe.all_channels(self.channelmap, self.electrodes)
        for e in self.probe.invalid_electrodes(self.channelmap, c, self.electrodes):
            e.state = ProbeDesp.STATE_FORBIDDEN
        for e in c:
            e.state = ProbeDesp.STATE_USED

    def update_electrode(self):
        """Refresh channelmap"""
        for state, data in self.data_electrodes.items():
            self.update_electrode_position(data, self.get_electrodes(None, state=state))
        self.update_electrode_position(self.data_highlight, [])

    def refresh_selection(self):
        """Rerun electrode selection and refresh channelmap"""
        self.logger.debug('refresh_selection()')
        t = time.time()
        try:
            self.channelmap = self.probe.select_electrodes(self.channelmap, self.electrodes, **self.selecting_parameters)

            t = time.time() - t
            self.logger.debug('refresh_selection() used %.2f sec', t)
        except BaseException as e:
            self.logger.warning('refresh_selection() fail', exc_info=e)
            raise  # raise error to ChannelMapEditorApp for logging message in message_area
        else:
            self._reset_electrode_state()

    def get_electrodes(self, s: None | int | list[int] | ColumnDataSource, *, state: int = None) -> list[E]:
        """
        Get electrodes in source *s*.

        :param s: electrode source.
        :param state: filter electrodes with its state in returns
        :return: electrodes
        """
        if (electrodes := self.electrodes) is None:
            return []

        ret: list[E]

        match s:
            case None:
                ret = list(electrodes)
            case int():
                ret = [electrodes[s]]
            case list():
                ret = [electrodes[it] for it in s]
            case _ if isinstance(s, ColumnDataSource):
                ret = [electrodes[it] for it in s.data['e']]
            case _:
                raise TypeError()

        if state is not None:
            ret = [it for it in ret if it.state == state]

        return ret

    def get_selected(self, d: ColumnDataSource = None, *, reset=False) -> set[E]:
        """
        Get selected electrodes.

        :param d: one of `self.data_electrodes`
        :param reset: reset the selecting state.
        :return: selected electrodes.
        """
        if d is None:
            ret = set[E]()
            for state, data in self.data_electrodes.items():
                ret.update(self.get_selected(data, reset=reset))
            return ret
        else:
            selected_index = d.selected.indices
            if reset:
                d.selected.indices = []

            e = d.data['e']
            return set(self.get_electrodes([e[it] for it in selected_index]))

    _on_select_electrode = []
    _on_select_callback = None

    def on_select(self, state: int):
        selected = self.get_selected(self.data_electrodes[state])
        self._on_select_electrode.extend(selected)
        if self._on_select_callback is None:
            self._on_select_callback = run_timeout(100, self._on_select)

    def _on_select(self):
        selected = self._on_select_electrode
        self._on_select_electrode = []
        self._on_select_callback = None

        if len(selected):
            self.logger.debug('select %d electrodes', len(selected))
        self.set_highlight(selected)

    def update_electrode_position(self, d: ColumnDataSource, e: Iterable[E]):
        """
        Show electrodes.

        :param d: one of `self.data_electrodes`
        :param e: new electrodes
        :param append: append *e*.
        """
        x = []
        y = []
        i = []
        c = []

        for it in e:
            x.append(it.x)
            y.append(it.y)
            i.append(self._e2i[it])
            c.append(str(it.channel))

        d.data = dict(x=x, y=y, e=i, c=c)

    def set_highlight(self, s: Iterable[E], *, invalid=True):
        """
        Highlight electrodes.

        :param s: selected electrode set
        :param invalid: extend the highlight set to include co-electrodes
        :param append: append the highlight set.
        """
        if invalid:
            s = self.probe.invalid_electrodes(self.channelmap, s, self.electrodes)

        self.update_electrode_position(self.data_highlight, s)

    def set_state_for_selected(self, state: int):
        """
        Set electrode state for selected electrodes.

        :param state: new state. value in {ProbeDesp.STATE_USED, ProbeDesp.STATE_UNUSED}
        """
        if state == ProbeDesp.STATE_USED:
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True):
                self.probe.add_electrode(self.channelmap, e)
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_FORBIDDEN], reset=True):
                self.probe.add_electrode(self.channelmap, e, overwrite=True)

            self.get_selected(self.data_electrodes[ProbeDesp.STATE_USED], reset=True)

        elif state == ProbeDesp.STATE_UNUSED:
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_USED], reset=True):
                self.probe.del_electrode(self.channelmap, e)

            self.get_selected(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True)
            self.get_selected(self.data_electrodes[ProbeDesp.STATE_FORBIDDEN], reset=True)

        self._reset_electrode_state()

    def set_policy_for_selected(self, policy: int):
        """
        Set electrode policy value for selected electrodes.

        :param policy: policy value from ProbeDesp.POLICY_*
        :return:
        """
        for e in self.get_selected(reset=True):
            e.policy = policy
