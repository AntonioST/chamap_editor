import argparse
import dataclasses
import logging
from pathlib import Path

from chmap.util.utils import doc_link

__all__ = [
    'ChannelMapEditorConfig',
    'new_parser',
    'parse_cli',
    'setup_logger',
]


@dataclasses.dataclass
@doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
class ChannelMapEditorConfig:
    """Start configuration for ChannelMapEditorApp.
    """

    # Source
    chmap_root: Path | None = None

    # Probe
    probe_family: str = 'npx'
    selector: str | None = None

    # Atlas
    atlas_name: int | str = 25
    atlas_root: Path | None = None

    # Application
    config_file: Path | None = None
    extra_view: list[str] = dataclasses.field(default_factory=list)
    server_address: str | None = None
    server_port: int | None = None
    no_open_browser: bool = False

    debug: bool = False

    # File
    open_file: str | None = None


@doc_link()
def new_parser() -> argparse.ArgumentParser:
    """Create a cli parse for {ChannelMapEditorConfig}."""
    ap = argparse.ArgumentParser(prog='chmap')

    ap.add_argument(metavar='FILE', nargs='?', type=Path, default=None, dest='open_file',
                    help='open channelmap file.')
    #
    gp = ap.add_argument_group('Source')
    gp.add_argument('-C', '--chmap-dir', metavar='PATH', type=Path, default=Path('.'), dest='chmap_root',
                    help='channel saving directory')

    #
    gp = ap.add_argument_group('Probe')
    gp.add_argument('-P', '--probe', metavar='NAME', default='npx', dest='probe_family',
                    help='use probe family. default use "npx" (Neuropixels probe family).')
    gp.add_argument('--selector', metavar='MODULE:NAME', default='default', dest='selector',
                    help='use which electrode selection method')

    #
    gp = ap.add_argument_group('Atlas')
    gp.add_argument('--atlas', metavar='NAME', type=_try_int, default=25, dest='atlas_name',
                    help='atlas mouse brain name')
    gp.add_argument('--atlas-root', metavar='PATH', type=Path, default=None, dest='atlas_root',
                    help='atlas mouse brain download path')

    #
    gp = ap.add_argument_group('Bokeh Application')
    gp.add_argument('--config-file', metavar='FILE', type=Path, default=None, dest='config_file',
                    help='global config file.')
    gp.add_argument('--view', metavar='MODULE:NAME', type=str, default=list(), action='append', dest='extra_view',
                    help='install extra views in right panel')
    gp.add_argument('--server-address', metavar='URL', default=None, dest='server_address',
                    help='')
    gp.add_argument('--server-port', metavar='PORT', type=int, default=None, dest='server_port',
                    help='')
    gp.add_argument('--no-open-browser', action='store_true', dest='no_open_browser',
                    help='do not open browser when server starts')
    gp.add_argument('--debug', action='store_true', dest='debug',
                    help=argparse.SUPPRESS)

    return ap


def parse_cli(args: list[str] = None) -> ChannelMapEditorConfig:
    """
    Parse command-line arguments and return result.

    :param args: command-line arguments list. use sys.argv if None.
    :return:
    """
    opt = new_parser().parse_args(args)
    kw = {}
    for field in dataclasses.fields(ChannelMapEditorConfig):
        kw[field.name] = getattr(opt, field.name)
    return ChannelMapEditorConfig(**kw)


def setup_logger(config: ChannelMapEditorConfig):
    """setup logger"""
    logging.basicConfig(
        format='[%(levelname)s] %(name)s - %(message)s'
    )

    if config.debug:
        logging.getLogger('chmap').setLevel(logging.DEBUG)


def _try_int(a: str) -> int | str:
    try:
        return int(a)
    except ValueError:
        return a


def _list_type(a: str) -> list[str]:
    return a.split(',')
