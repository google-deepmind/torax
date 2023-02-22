# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic post-run plotting tool. Plot a single run or comparison of two runs.

Includes a time slider. Reads output h5 files,

Plots:
(1) chi_i, chi_e (transport coefficients)
(2) Ti, Te (temperatures)
(3) ne (density)
(4) jtot, johm (total and ohmic plasma current)
(5) q (safety factor)
(6) s (magnetic shear)
"""

import argparse
import dataclasses
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # pylint: disable=g-importing-member
import numpy as np

matplotlib.use('TkAgg')


@dataclasses.dataclass
class PlotData:
  """Dataclass for all plot related data."""

  ti: np.ndarray
  te: np.ndarray
  ne: np.ndarray
  j: np.ndarray
  johm: np.ndarray
  q: np.ndarray
  s: np.ndarray
  t: np.ndarray
  rcell_coord: np.ndarray
  rface_coord: np.ndarray

  def __post_init__(self):
    self.tmin = min(self.t)
    self.tmax = max(self.t)
    self.ymax_t = np.amax([self.ti, self.te])
    self.ymax_n = np.amax(self.ne)
    self.ymax_j = np.amax([np.amax(self.j), np.amax(self.johm)])
    self.ymin_j = np.amin([np.amin(self.j), np.amin(self.johm)])
    self.ymin_j = np.amin(self.j)
    self.ymax_q = np.amax(self.q)
    self.ymax_s = np.amax(self.s)
    self.ymin_s = np.amin(self.s)
    self.dt = min(np.diff(self.t))


parser = argparse.ArgumentParser(description='Plot finished run')
parser.add_argument(
    '--outfile',
    nargs='*',
    help=(
        'Relative location of output h5 files (if two are provided, a'
        ' comparison is done)'
    ),
)
parser.set_defaults(normalized=True)

args = parser.parse_args()
if (
    len(args.outfile) > 1
):  # only the second argument will be processed for the comparison plots
  comp_plot = True
else:
  comp_plot = False

if not args.outfile:
  raise ValueError('No output file provided')

with h5py.File(args.outfile[0] + 'state_history.h5', 'r') as hf:
  plotdata = PlotData(
      ti=hf['temp_ion'][:],
      te=hf['temp_el'][:],
      ne=hf['ne'][:],
      j=hf['jtot'][:],
      johm=hf['johm'][:],
      q=hf['q_face'][:],
      s=hf['s_face'][:],
      t=hf['t'][:],
      rcell_coord=hf['r_cell_norm'][:],
      rface_coord=hf['r_face_norm'][:],
  )

if comp_plot:
  with h5py.File(args.outfile[1] + 'state_history.h5', 'r') as hf:
    plotdata2 = PlotData(
        ti=hf['temp_ion'][:],
        te=hf['temp_el'][:],
        ne=hf['ne'][:],
        j=hf['jtot'][:],
        johm=hf['johm'][:],
        q=hf['q_face'][:],
        s=hf['s_face'][:],
        t=hf['t'][:],
        rcell_coord=hf['r_cell_norm'][:],
        rface_coord=hf['r_face_norm'][:],
    )

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

lines = []
lines2 = []

# TODO(b/323504363): improve efficiency through use of line plotting functions
if comp_plot:
  ax1.set_title('(1)=' + args.outfile[0] + ', (2)=' + args.outfile[1])
else:
  ax1.set_title('(1)=' + args.outfile[0])
(line,) = ax2.plot(
    plotdata.rcell_coord, plotdata.ti[0, :], 'r', label=r'$T_i~(1)$'
)
lines.append(line)
(line,) = ax2.plot(
    plotdata.rcell_coord, plotdata.te[0, :], 'b', label=r'$T_e~(1)$'
)
lines.append(line)
(line,) = ax3.plot(
    plotdata.rcell_coord, plotdata.ne[0, :], 'r', label=r'$n_e~(1)$'
)
lines.append(line)

(line,) = ax4.plot(
    plotdata.rcell_coord, plotdata.j[0, :], 'r', label=r'$j_{tot}~(1)$'
)
lines.append(line)
(line,) = ax4.plot(
    plotdata.rcell_coord, plotdata.johm[0, :], 'b', label=r'$j_{ohm}~(1)$'
)
lines.append(line)
(line,) = ax5.plot(
    plotdata.rface_coord, plotdata.q[0, :], 'r', label=r'$q~(1)$'
)
lines.append(line)
(line,) = ax6.plot(
    plotdata.rface_coord, plotdata.s[0, :], 'r', label=r'$\hat{s}~(1)$'
)
lines.append(line)

# pylint: disable=undefined-variable
if comp_plot:
  (line,) = ax2.plot(
      plotdata2.rcell_coord, plotdata2.ti[0, :], 'r--', label=r'$T_i (2)$'
  )
  lines2.append(line)
  (line,) = ax2.plot(
      plotdata2.rcell_coord, plotdata2.te[0, :], 'b--', label=r'$T_e (2)$'
  )
  lines2.append(line)
  (line,) = ax3.plot(
      plotdata2.rcell_coord, plotdata2.ne[0, :], 'r--', label=r'$n_e (2)$'
  )
  lines2.append(line)
  (line,) = ax4.plot(
      plotdata2.rcell_coord, plotdata2.j[0, :], 'r--', label=r'$j_{tot} (2)$'
  )
  lines2.append(line)
  (line,) = ax4.plot(
      plotdata2.rcell_coord, plotdata2.johm[0, :], 'b--', label=r'$j_{ohm} (2)$'
  )
  lines2.append(line)
  (line,) = ax5.plot(
      plotdata2.rface_coord, plotdata2.q[0, :], 'r--', label=r'$q (2)$'
  )
  lines2.append(line)
  (line,) = ax6.plot(
      plotdata2.rface_coord, plotdata2.s[0, :], 'r--', label=r'$\hat{s} (2)$'
  )
  lines2.append(line)
# pylint: enable=undefined-variable

# TODO(b/323504363): add heat conductivity to output h5
ax1.set_xlabel('Normalized radius')
ax1.set_ylabel(r'Heat conductivity $[m^2/s]$')
ax1.legend()

ax2.set_ylim([0, plotdata.ymax_t * 1.05])
ax2.set_xlabel('Normalized radius')
ax2.set_ylabel('Temperature [keV]')
ax2.legend()

ax3.set_ylim([0, plotdata.ymax_n * 1.05])
ax3.set_xlabel('Normalized radius')
ax3.set_ylabel(r'Electron density $[10^{20}~m^{-3}]$')
ax3.legend()


ax4.set_ylim([min(plotdata.ymin_j * 1.05, 0), plotdata.ymax_j * 1.05])
ax4.set_xlabel('Normalized radius')
ax4.set_ylabel(r'Toroidal current $[A~m^{-2}]$')
ax4.legend()

ax5.set_ylim([0, plotdata.ymax_q * 1.05])
ax5.set_xlabel('Normalized radius')
ax5.set_ylabel('Safety factor')
ax5.legend()

ax6.set_ylim([min(plotdata.ymin_s * 1.05, 0), plotdata.ymax_s * 1.05])
ax6.set_xlabel('Normalized radius')
ax6.set_ylabel('Magnetic shear')
ax6.legend()


plt.subplots_adjust(bottom=0.2)
axslide = plt.axes([0.12, 0.05, 0.75, 0.05])

# pylint: disable=undefined-variable
if comp_plot:
  dt = min(plotdata.dt, plotdata2.dt)
else:
  dt = plotdata.dt
# pylint: enable=undefined-variable

timeslider = Slider(
    axslide,
    'Time [s]',
    plotdata.tmin,
    plotdata.tmax,
    valinit=plotdata.tmin,
    valstep=dt,
)

fig.canvas.draw()


def update(newtime):
  """Update plots with new values following slider manipulation."""
  idx = np.abs(plotdata.t - newtime).argmin()  # find index closest to new time
  datalist = [
      plotdata.ti[idx, :],
      plotdata.te[idx, :],
      plotdata.ne[idx, :],
      plotdata.j[idx, :],
      plotdata.johm[idx, :],
      plotdata.q[idx, :],
      plotdata.s[idx, :],
  ]
  for plotline1, data in zip(lines, datalist):
    plotline1.set_ydata(data)
  if comp_plot:
    idx = np.abs(
        plotdata2.t - newtime
    ).argmin()  # find index closest to new time
    datalist2 = [
        plotdata2.ti[idx, :],
        plotdata2.te[idx, :],
        plotdata2.ne[idx, :],
        plotdata2.j[idx, :],
        plotdata2.johm[idx, :],
        plotdata2.q[idx, :],
        plotdata2.s[idx, :],
    ]
    for plotline2, data in zip(lines2, datalist2):
      plotline2.set_ydata(data)
  fig.canvas.draw()


# Call update function when slider value is changed
timeslider.on_changed(update)
plt.show()

# fig.tight_layout()
