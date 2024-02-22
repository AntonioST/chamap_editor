from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from bokeh.models import DataTable, ColumnDataSource, TableColumn, TextInput, Div, CDSView, CustomJS, Toggle, Button, AutocompleteInput
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.files import user_cache_file
from chmap.util.bokeh_util import ButtonFactory, as_callback, is_recursive_called
from .base import RecordStep, RecordView, R, ViewBase, ControllerView, InvisibleView

__all__ = ['RecordManager', 'HistoryView']


class NamedHistory(NamedTuple):
    name: str
    frozen: bool
    steps: list[RecordStep]

    @property
    def size(self) -> int:
        return len(self.steps)

    def append(self, step: RecordStep | NamedHistory) -> bool:
        if not self.frozen:
            if isinstance(step, NamedHistory):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)
            return True
        return False

    def filter(self, index: list[int]) -> list[RecordStep]:
        return [self.steps[it] for it in index]

    def delete(self, index: list[int]) -> bool:
        if not self.frozen:
            for i in sorted(index, reverse=True):
                del self.steps[i]
            return True
        return False

    def clear(self) -> bool:
        if not self.frozen:
            self.steps.clear()
            return True
        return False


class RecordManager:
    def __init__(self):
        self.logger = logging.getLogger('chmap.history')

        self.views: list[RecordView] = []
        self._history: dict[str, NamedHistory] = {}
        self.name = ''
        self.disable = False
        self._is_replaying = False
        self._view: HistoryView | None = None

    # ======= #
    # history #
    # ======= #

    def list_history_names(self) -> list[str]:
        return list(self._history.keys())

    @property
    def history(self) -> NamedHistory:
        try:
            return self._history[self.name]
        except KeyError:
            pass

        ret = NamedHistory(self.name, False, [])
        self._history[ret.name] = ret
        return ret

    def has_history(self, name: str) -> bool:
        return name in self._history

    def get_history(self, name: str) -> NamedHistory:
        return self._history[name]

    def clear_history(self, name: str = None) -> bool:
        """
        Clear history and remove it.

        Do nothing if this history is frozen.

        :param name: history name.
        :return: cleared and removed.
        """
        if name is None:
            name = self.name

        if (history := self._history.get(name, None)) is not None and history.clear():
            del self._history[name]
            return True

        return False

    # ============= #
    # view register #
    # ============= #

    def register(self, view: RecordView):
        if view in self.views:
            return

        self.logger.debug('register %s', type(view).__name__)

        def add_record(record: R, category: str, description: str):
            if not self._is_replaying and not self.disable:
                self._add_record(view, record, category, description)

        setattr(view, 'add_record', add_record)
        self.views.append(view)

    def unregister(self, view: RecordView):
        try:
            i = self.views.index(view)
        except ValueError:
            return
        else:
            self.logger.debug('unregister %s', type(view).__name__)
            del self.views[i]
            setattr(view, 'add_record', RecordView.add_record)

    # ============= #
    # record/replay #
    # ============= #

    def _add_record(self, view: RecordView[R], record: R, category: str, description: str):
        step = RecordStep(type(view).__name__, time.time(), category, description, record)
        self.logger.debug('get %s', step)
        if self.history.append(step) and (history := self._view) is not None:
            history.on_add_record(step)

    def replay(self, name: str = None, *, reset=False, index: list[int] = None):
        """
        replay history.

        :param name: history name
        :param reset: Is it a reset replay?
        :param index: only replay steps on given index.
        :raise KeyError: history *name* not exist.
        """
        if self._is_replaying:
            raise RuntimeError()

        if name is None:
            history = self.history
            name = history.name
        else:
            history = self._history[name]

        if index is None:
            self.logger.debug('replay history[%s] all steps', name)
            steps = list(history.steps)
        else:
            self.logger.debug('replay history[%s] steps on %s', name, index)
            steps = history.filter(index)

        self._is_replaying = True
        try:
            for view in self.views:
                view.replay_records(steps, reset=reset)
        finally:
            self._is_replaying = False

    # ================= #
    # save/load history #
    # ================= #

    def load_history(self, file: str | Path, *, reset=False):
        """

        :param file: history json file.
        :param reset: reset history. Otherwise, update it.
        :return: {name: list[RecordStep]}
        """
        self.logger.debug('load history from %s', file)
        import json
        with Path(file).open() as f:
            data = json.load(f)

        ret: dict[str, NamedHistory] = {}
        for name, history in data.items():
            steps = []
            for item in history.get('steps', []):
                steps.append(RecordStep.from_dict(item))

            frozen = history.get('frozen', False)
            ret[name] = NamedHistory(name, frozen, steps)

        if reset:
            self._history = ret
        else:
            self._history.update(ret)

    def save_history(self, file: str | Path):
        self.logger.debug('save history %s', file)

        import json

        data = {
            name: dict(
                frozen=history.frozen,
                steps=[it.as_dict() for it in history.steps]
            )
            for name, history in self._history.items()
        }

        with Path(file).open('w') as f:
            json.dump(data, f, indent=2)


class HistoryView(ViewBase, ControllerView, InvisibleView):
    history_step_data: ColumnDataSource
    history_step_view: CDSView

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.history')
        self._config = config
        self.manager: RecordManager | None = None
        self.history_step_data = ColumnDataSource(data=dict(source=[], category=[], action=[]))
        self.history_step_view = CDSView()

    @property
    def name(self) -> str:
        return 'History'

    def get_history_file(self) -> Path:
        return user_cache_file(self._config, 'history.json')

    def load_history(self):
        if (manager := self.manager) is not None and (history_file := self.get_history_file()).exists():
            self.log_message('load history')
            manager.load_history(history_file)
            manager.name = self.save_input.value
            self.update_history_table()

    def save_history(self):
        if (manager := self.manager) is not None:
            self.log_message('save history')
            history_file = self.get_history_file()
            history_file.parent.mkdir(parents=True, exist_ok=True)
            manager.save_history(history_file)
            self.logger.debug(f'save history : %s', history_file)

    def on_add_record(self, record: RecordStep):
        self.history_step_data.stream(dict(
            source=[record.source],
            category=[record.category],
            action=[record.description]
        ))

    def update_history_table(self):
        if is_recursive_called():
            return

        if (manager := self.manager) is None:
            self.history_step_data.data = dict(source=[], category=[], action=[])
            return

        self.save_input.completions = names = manager.list_history_names()
        if len(self.save_input.value) == 0 and len(names) > 0:
            self.save_input.value = names[0]

        source = []
        category = []
        action = []
        for i, record in enumerate(manager.history.steps):
            source.append(record.source)
            category.append(record.category)
            action.append(record.description)

        self.logger.debug('update history table')
        self.history_step_data.data = dict(source=source, category=category, action=action)

    # ============= #
    # UI components #
    # ============= #

    history_step_table: DataTable
    source_filter: TextInput
    category_filter: TextInput
    disable_toggle: Toggle
    description_filter: TextInput
    save_input: AutocompleteInput
    delete_btn: Button
    clear_btn: Button

    def _setup_content(self, **kwargs):
        new_btn = ButtonFactory(min_width=100, width_policy='min')

        # history table
        self.history_step_table = DataTable(
            source=self.history_step_data,
            view=self.history_step_view,
            columns=[
                TableColumn(field='source', title='Source', width=100),
                TableColumn(field='category', title='Category', width=100),
                TableColumn(field='action', title='Action'),
            ],
            width=500, height=300, reorderable=False, sortable=False,
        )

        # filter inputs
        d1 = "Match exactly. Use ',' to select multiple"
        d2 = "Match Any. Use '~', '&' or '|' to do selection inverse, intersect or union, respectively."
        self.source_filter = TextInput(title='Source', width=100, description=d1)
        self.source_filter.on_change('value', as_callback(self._on_filter_update))
        self.category_filter = TextInput(title='Category', width=100, description=d1)
        self.category_filter.on_change('value', as_callback(self._on_filter_update))
        self.description_filter = TextInput(title='Action', width=200, description=d2)
        self.description_filter.on_change('value', as_callback(self._on_filter_update))

        # save input
        self.save_input = AutocompleteInput(
            title='Name', width=100, description='History name',
            min_characters=0, max_completions=5, case_sensitive=False, restrict=False,
        )
        self.save_input.on_change('value', as_callback(self._on_save_name_change))

        # buttons
        self.delete_btn = new_btn('Delete', self.on_delete)
        self.clear_btn = new_btn('Clear', self.on_clear)

        # buttons - replay
        replay_callback = TextInput(visible=False)
        replay_callback.on_change('value', as_callback(self._on_replay))

        # https://discourse.bokeh.org/t/get-the-number-of-elements-returned-by-a-cdsview-in-bokeh/11206/4
        replay = new_btn('Replay', None)
        replay.js_on_click(handler=CustomJS(args=dict(view=self.history_step_view, callback=replay_callback), code="""
        callback.value = view._indices.join(',');
        """))

        # buttons
        self.disable_toggle = Toggle(label='Disable', min_width=100, width_policy='min')
        self.disable_toggle.on_change('active', as_callback(self._on_disable))

        from bokeh.layouts import row, column
        return [
            row(
                column(
                    row(Div(text='<b>Filter</b>'), self.source_filter, self.category_filter, self.description_filter),
                    self.history_step_table,

                ),
                column(
                    self.save_input,
                    replay,
                    new_btn('Save', self.save_history),
                    new_btn('Load', self.load_history),
                    self.delete_btn,
                    self.clear_btn,
                    self.disable_toggle,
                    replay_callback
                )
            )
        ]

    def _on_filter_update(self):
        from bokeh.models import filters

        value: str
        selectors = []
        if len(value := self.source_filter.value.strip()) > 0:
            selectors.append(self._on_filter_category('source', value))

        if len(value := self.category_filter.value.strip()) > 0:
            selectors.append(self._on_filter_category('category', value))

        if len(value := self.description_filter.value.strip()) > 0:
            description = list(self.history_step_data.data['action'])
            booleans = self._on_filter_description(description, value)
            selectors.append(filters.BooleanFilter(booleans=list(booleans)))

        if len(selectors) == 0:
            self.history_step_view.filter = filters.AllIndices()
        elif len(selectors) == 1:
            self.history_step_view.filter = selectors[0]
        else:
            self.history_step_view.filter = filters.IntersectionFilter(operands=selectors)

    def _on_filter_category(self, column: str, value: str):
        from bokeh.models import filters

        if ',' in value:
            return filters.UnionFilter(operands=[
                filters.GroupFilter(column_name=column, group=it.strip())
                for it in value.split(',')
            ])
        else:
            return filters.GroupFilter(column_name=column, group=value)

    def _on_filter_description(self, description: list[str], expr: str) -> NDArray[np.bool_]:
        old_expr = expr
        expr = re.sub(r'([^ \t~|&^()]+)', r'_a("\1")', expr)
        self.logger.debug('filter description "%s" -> "%s"', old_expr, expr)

        def _a(word: str) -> NDArray[np.bool_]:
            word = word.lower()
            return np.array([word in it.lower() for it in description])

        return eval(expr, {}, dict(_a=_a))

    def _on_save_name_change(self, old: str, name: str):
        if (manager := self.manager) is not None:
            if len(name) == 0:
                self.log_message(f'change default history')
            else:
                self.log_message(f'change history {name}')

            copy = not manager.has_history(name)

            manager.name = name

            history = manager.history
            if history.frozen:
                self.delete_btn.disabled = True
                self.clear_btn.disabled = True
            else:
                self.delete_btn.disabled = False
                self.clear_btn.disabled = False

            if copy:
                try:
                    from_history = manager.get_history(old)
                except KeyError:
                    pass
                else:
                    self.logger.debug('copy history from %s', old)
                    history.append(from_history)

            self.update_history_table()

    def _on_replay(self, value: str):
        if (manager := self.manager) is None:
            return

        self.logger.debug('replay')
        index = list(map(int, value.split(',')))

        manager.replay(reset=True, index=index)
        self.get_app().on_probe_update()

    def on_replay(self):
        if (manager := self.manager) is None:
            return

        self.logger.debug('replay')
        manager.replay(reset=True)
        self.get_app().on_probe_update()

    def on_delete(self):
        selected = list(self.history_step_data.selected.indices)

        if (manager := self.manager) is not None:
            self.logger.debug('delete')
            if manager.history.delete(selected):
                self.update_history_table()

    def on_clear(self):
        if (manager := self.manager) is not None:
            self.logger.debug('clear')

            is_empty = manager.history.size == 0

            if manager.clear_history():
                if is_empty:
                    self.logger.debug('remove')
                    self.save_input.value = ''
                else:
                    self.update_history_table()

    def _on_disable(self, disable: bool):
        if (manager := self.manager) is not None:
            manager.disable = disable

    # ============== #
    # update methods #
    # ============== #

    def start(self):
        self.manager = self.get_app().record_manager
        if self.manager is None:
            self.log_message('app history feature is disabled')
        else:
            self.manager._view = self
            self.update_history_table()


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.record:HistoryView',
    ]))
