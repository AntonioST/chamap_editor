import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from chmap.probe_npx.desp import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.probe_npx.plot import plot_probe_shape, plot_channelmap_block
from chmap.probe_npx.select import electrode_select

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path(sys.argv[1])
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
policy = np.load(file.with_suffix('.policy.npy'))
electrodes = desp.electrode_from_numpy(desp.all_electrodes(chmap), policy)
chmap = electrode_select(desp, chmap, electrodes, selector='weaker')

fg, ax = plt.subplots()
height = 6

plot_channelmap_block(ax, chmap, height=height, color='k', shank_width_scale=2)
plot_probe_shape(ax, chmap.probe_type, height=height, color='gray', label_axis=True, shank_width_scale=2)

if len(sys.argv) == 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])