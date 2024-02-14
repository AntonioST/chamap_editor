from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, TypeVar

from chmap.util.utils import import_name
from .base import ViewBase, StateView, ControllerView, EditorView, GlobalStateView

if TYPE_CHECKING:
    from chmap.config import ChannelMapEditorConfig
    from chmap.main_bokeh import ChannelMapEditorApp

__all__ = ['init_view']

V = TypeVar('V', bound=ViewBase)


def init_view(config: ChannelMapEditorConfig, view_type) -> ViewBase | None:
    """

    Recognised type:

    * `None` skip
    * `ViewBase` or `type[ViewBase]`
    * `ImageHandler` or `type[ImageHandler]`, wrap with ImageView.
    * literal 'file' for FileImageView
    * literal 'atlas' for AtlasBrainView
    * literal 'blueprint' for BlueprintView
    * literal 'editor' for InitializeBlueprintView
    * image filepath
    * `str` in pattern: `module.path:attribute` in type listed above.

    :param config:
    :param view_type:
    :return:
    """
    from chmap.views.image import ImageView, ImageHandler

    try:
        if isinstance(view_type, type) and issubclass(view_type, ViewBase):
            return view_type(config)

        elif isinstance(view_type, ViewBase):
            return view_type

        elif isinstance(view_type, type) and issubclass(view_type, ImageHandler):
            return ImageView(config, view_type())

        elif isinstance(view_type, ImageHandler):
            return ImageView(config, view_type)

        elif view_type == 'file':
            from .image import FileImageView
            return FileImageView(config)

        elif view_type == 'atlas':
            from .atlas import AtlasBrainView
            return AtlasBrainView(config)

        elif view_type == 'blueprint':
            from .blueprint import BlueprintView
            return BlueprintView(config)

        elif view_type == 'editor':
            from .edit_blueprint import InitializeBlueprintView
            return InitializeBlueprintView(config)

        elif isinstance(view_type, str) and is_image(image_file := Path(view_type)):
            from chmap.views.image import ImageView, ImageHandler
            return ImageView(config, ImageHandler.from_file(image_file))

        elif isinstance(view_type, str):
            return import_view(config, view_type)
        else:
            raise RuntimeError(f'unknown view_type : {view_type}')

    except BaseException as e:
        logging.getLogger('chmap.view').warning('init view fail', exc_info=e)
        pass

    return None


def import_view(config: ChannelMapEditorConfig, module_path: str) -> ViewBase | None:
    logging.getLogger('chmap.view').debug('import %s', module_path)
    return init_view(config, import_name('view base', module_path))


def install_view(app: ChannelMapEditorApp, view: V) -> V:
    """
    Replace some methods in ViewBase. They are

    * ViewBase.log_message
    * ControllerView.*
    * EditorView.*
    * GlobalStateView.*

    :param app:
    :param view:
    :return:
    """

    def log_message(*message, reset=False):
        app.log_message(*message, reset=reset)

    def get_app() -> ChannelMapEditorApp:
        return app

    def get_view(view_type: str | type[ViewBase]) -> ViewBase | None:
        for _view in app.right_panel_views:
            if isinstance(view_type, type) and isinstance(_view, view_type):
                return _view
            elif isinstance(view_type, str) and type(_view).__name__ == view_type:
                return _view
        return None

    def update_probe():
        app.logger.debug('update_probe(%s)', type(view).__name__)
        app.on_probe_update()

    def save_global_state(state=None, *, sync=False, force=False):
        if not getattr(view, 'disable_save_global_state', False) or force:
            app.logger.debug('save_global_state(%s)', type(view).__name__)

            if sync:
                app.save_global_config(direct=False)
            else:
                if state is None:
                    state = cast(StateView, view).save_state()
                app.global_views_config[type(view).__name__] = state
                app.save_global_config(direct=True)

    def restore_global_state(*, reload=False, force=False):
        app.logger.debug('restore_global_state(%s)', type(view).__name__)

        if reload:
            app.load_global_config(reset=False)

        try:
            config = app.global_views_config[type(view).__name__]
        except KeyError:
            if force:
                cast(StateView, view).restore_state({})
        else:
            cast(StateView, view).restore_state(config)

    setattr(view, 'log_message', log_message)

    if isinstance(view, ControllerView):
        setattr(view, 'get_app', get_app)
        setattr(view, 'get_view', get_view)

    if isinstance(view, EditorView):
        setattr(view, 'update_probe', update_probe)

    if isinstance(view, GlobalStateView):
        setattr(view, 'save_global_state', save_global_state)
        setattr(view, 'restore_global_state', restore_global_state)

    return view


def is_image(path: Path) -> bool:
    if not path.is_file():
        return False

    import mimetypes
    mt, _ = mimetypes.guess_type(path)
    return mt is not None and mt.startswith('image/')