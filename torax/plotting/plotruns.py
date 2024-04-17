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

Includes a time slider. Reads output files with xarray data or legacy h5 data.

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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # pylint: disable=g-importing-member
import numpy as np
import xarray as xr

matplotlib.use('TkAgg')


@dataclasses.dataclass
class PlotData:
  """Dataclass for all plot related data."""

  ti: np.ndarray
  te: np.ndarray
  ne: np.ndarray
  j: np.ndarray
  johm: np.ndarray
  j_bootstrap: np.ndarray
  jext: np.ndarray
  q: np.ndarray
  s: np.ndarray
  chi_i: np.ndarray
  chi_e: np.ndarray
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
    # avoid initial condition for chi ymax, since can be unphysically high
    self.ymax_chi_i = np.amax(self.chi_i[1:, :])
    self.ymax_chi_e = np.amax(self.chi_e[1:, :])
    self.dt = min(np.diff(self.t))


parser = argparse.ArgumentParser(description='Plot finished run')
parser.add_argument(
    '--outfile',
    nargs='*',
    help=(
        'Relative location of output files (if two are provided, a'
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

ds1 = xr.open_dataset(args.outfile[0])
if 'time' in ds1:
  t = ds1['time'].to_numpy()
else:
  t = ds1['t'].to_numpy()
plotdata1 = PlotData(
    ti=ds1['temp_ion'].to_numpy(),
    te=ds1['temp_el'].to_numpy(),
    ne=ds1['ne'].to_numpy(),
    j=ds1['jtot'].to_numpy(),
    johm=ds1['johm'].to_numpy(),
    j_bootstrap=ds1['j_bootstrap'].to_numpy(),
    jext=ds1['jext'].to_numpy(),
    q=ds1['q_face'].to_numpy(),
    s=ds1['s_face'].to_numpy(),
    chi_i=ds1['chi_face_ion'].to_numpy(),
    chi_e=ds1['chi_face_el'].to_numpy(),
    rcell_coord=ds1['r_cell_norm'].to_numpy(),
    rface_coord=ds1['r_face_norm'].to_numpy(),
    t=t,
)

if comp_plot:
  ds2 = xr.open_dataset(args.outfile[1])
  if 'time' in ds2:
    t = ds2['time'].to_numpy()
  else:
    t = ds2['t'].to_numpy()
  plotdata2 = PlotData(
      ti=ds2['temp_ion'].to_numpy(),
      te=ds2['temp_el'].to_numpy(),
      ne=ds2['ne'].to_numpy(),
      j=ds2['jtot'].to_numpy(),
      johm=ds2['johm'].to_numpy(),
      j_bootstrap=ds2['j_bootstrap'].to_numpy(),
      jext=ds2['jext'].to_numpy(),
      q=ds2['q_face'].to_numpy(),
      s=ds2['s_face'].to_numpy(),
      chi_i=ds2['chi_face_ion'].to_numpy(),
      chi_e=ds2['chi_face_el'].to_numpy(),
      rcell_coord=ds2['r_cell_norm'].to_numpy(),
      rface_coord=ds2['r_face_norm'].to_numpy(),
      t=t,
  )

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

lines1 = []
lines2 = []

if comp_plot:
  ax2.set_title('(1)=' + args.outfile[0] + ', (2)=' + args.outfile[1])
else:
  ax2.set_title('(1)=' + args.outfile[0])
(line,) = ax1.plot(
    plotdata1.rface_coord, plotdata1.chi_i[1, :], 'r', label=r'$\chi_i~(1)$'
)
lines1.append(line)
(line,) = ax1.plot(
    plotdata1.rface_coord, plotdata1.chi_e[1, :], 'b', label=r'$\chi_e~(1)$'
)
lines1.append(line)
(line,) = ax2.plot(
    plotdata1.rcell_coord, plotdata1.ti[0, :], 'r', label=r'$T_i~(1)$'
)
lines1.append(line)
(line,) = ax2.plot(
    plotdata1.rcell_coord, plotdata1.te[0, :], 'b', label=r'$T_e~(1)$'
)
lines1.append(line)
(line,) = ax3.plot(
    plotdata1.rcell_coord, plotdata1.ne[0, :], 'r', label=r'$n_e~(1)$'
)
lines1.append(line)

(line,) = ax4.plot(
    plotdata1.rcell_coord, plotdata1.j[0, :], 'r', label=r'$j_{tot}~(1)$'
)
lines1.append(line)
(line,) = ax4.plot(
    plotdata1.rcell_coord, plotdata1.johm[0, :], 'b', label=r'$j_{ohm}~(1)$'
)
lines1.append(line)
(line,) = ax4.plot(
    plotdata1.rcell_coord,
    plotdata1.j_bootstrap[0, :],
    'g',
    label=r'$j_{bs}~(1)$',
)
lines1.append(line)
(line,) = ax4.plot(
    plotdata1.rcell_coord, plotdata1.jext[0, :], 'm', label=r'$j_{ext}~(1)$'
)
lines1.append(line)
(line,) = ax5.plot(
    plotdata1.rface_coord, plotdata1.q[0, :], 'r', label=r'$q~(1)$'
)
lines1.append(line)
(line,) = ax6.plot(
    plotdata1.rface_coord, plotdata1.s[0, :], 'r', label=r'$\hat{s}~(1)$'
)
lines1.append(line)

# pylint: disable=undefined-variable
if comp_plot:
  (line,) = ax1.plot(
      plotdata2.rface_coord, plotdata2.chi_i[1, :], 'r--', label=r'$\chi_i~(2)$'
  )
  lines2.append(line)
  (line,) = ax1.plot(
      plotdata2.rface_coord, plotdata2.chi_e[1, :], 'b--', label=r'$\chi_e~(2)$'
  )
  lines2.append(line)
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
  (line,) = ax4.plot(
      plotdata2.rcell_coord,
      plotdata2.j_bootstrap[0, :],
      'g',
      label=r'$j_{bs}~(2)$',
  )
  lines2.append(line)
  (line,) = ax4.plot(
      plotdata2.rcell_coord, plotdata2.jext[0, :], 'm', label=r'$j_{ext}~(2)$'
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

# ax1.set_ylim([0, np.max([plotdata1.ymax_chi_i, plotdata1.ymax_chi_e]) * 1.05])
ax1.set_xlabel('Normalized radius')
ax1.set_ylabel(r'Heat conductivity $[m^2/s]$')
ax1.legend()

ax2.set_ylim([0, plotdata1.ymax_t * 1.05])
ax2.set_xlabel('Normalized radius')
ax2.set_ylabel('Temperature [keV]')
ax2.legend()

ax3.set_ylim([0, plotdata1.ymax_n * 1.05])
ax3.set_xlabel('Normalized radius')
ax3.set_ylabel(r'Electron density $[10^{20}~m^{-3}]$')
ax3.legend()


ax4.set_ylim([min(plotdata1.ymin_j * 1.05, 0), plotdata1.ymax_j * 1.05])
ax4.set_xlabel('Normalized radius')
ax4.set_ylabel(r'Toroidal current $[A~m^{-2}]$')
ax4.legend(fontsize=10)

ax5.set_ylim([0, plotdata1.ymax_q * 1.05])
ax5.set_xlabel('Normalized radius')
ax5.set_ylabel('Safety factor')
ax5.legend()

ax6.set_ylim([min(plotdata1.ymin_s * 1.05, 0), plotdata1.ymax_s * 1.05])
ax6.set_xlabel('Normalized radius')
ax6.set_ylabel('Magnetic shear')
ax6.legend()


plt.subplots_adjust(bottom=0.2)
axslide = plt.axes([0.12, 0.05, 0.75, 0.05])

# pylint: disable=undefined-variable
if comp_plot:
  dt = min(plotdata1.dt, plotdata2.dt)
else:
  dt = plotdata1.dt
# pylint: enable=undefined-variable

timeslider = Slider(
    axslide,
    'Time [s]',
    plotdata1.tmin,
    plotdata1.tmax,
    valinit=plotdata1.tmin,
    valstep=dt,
)

fig.canvas.draw()


def update(newtime):
  """Update plots with new values following slider manipulation."""
  idx = np.abs(plotdata1.t - newtime).argmin()  # find index closest to new time
  datalist1 = [
      plotdata1.chi_i[idx, :],
      plotdata1.chi_e[idx, :],
      plotdata1.ti[idx, :],
      plotdata1.te[idx, :],
      plotdata1.ne[idx, :],
      plotdata1.j[idx, :],
      plotdata1.johm[idx, :],
      plotdata1.j_bootstrap[idx, :],
      plotdata1.jext[idx, :],
      plotdata1.q[idx, :],
      plotdata1.s[idx, :],
  ]
  for plotline1, data1 in zip(lines1, datalist1):
    plotline1.set_ydata(data1)
  if comp_plot:
    idx = np.abs(
        plotdata2.t - newtime
    ).argmin()  # find index closest to new time
    datalist2 = [
        plotdata2.chi_i[idx, :],
        plotdata2.chi_e[idx, :],
        plotdata2.ti[idx, :],
        plotdata2.te[idx, :],
        plotdata2.ne[idx, :],
        plotdata2.j[idx, :],
        plotdata2.johm[idx, :],
        plotdata2.j_bootstrap[idx, :],
        plotdata2.jext[idx, :],
        plotdata2.q[idx, :],
        plotdata2.s[idx, :],
    ]
    for plotline2, data2 in zip(lines2, datalist2):
      plotline2.set_ydata(data2)
  fig.canvas.draw()


# Call update function when slider value is changed
timeslider.on_changed(update)
plt.show()

fig.tight_layout()
