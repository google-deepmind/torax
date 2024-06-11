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

"""Utilities for plotting outputs of Torax runs."""

from collections.abc import Sequence
import dataclasses
import functools
from os import path
from typing import Any

import matplotlib
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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


def plot_run(outfile: str, outfile2: str | None = None):
  """Plots a single run or comparison of two runs."""
  filename1, filename2 = outfile, outfile2
  if not path.exists(outfile):
    raise ValueError(f'File {outfile} does not exist.')
  if outfile2 is not None and not path.exists(outfile2):
    raise ValueError(f'File {outfile2} does not exist.')
  plotdata1 = load_data(outfile)
  plotdata2 = None
  if outfile2 is not None:
    plotdata2 = load_data(outfile2)

  fig, subfigures = create_figure()
  ax2 = subfigures[1]
  if outfile2 is not None:
    ax2.set_title('(1)=' + filename1 + ', (2)=' + filename2)
  else:
    ax2.set_title('(1)=' + filename1)

  lines1 = get_lines(
      plotdata1,
      subfigures,
  )
  lines2 = None
  if plotdata2 is not None:
    lines2 = get_lines(plotdata2, subfigures, comp_plot=True)

  format_plots(plotdata1, subfigures)
  timeslider = create_slider(plotdata1, plotdata2)
  fig.canvas.draw()

  update = functools.partial(
      _update,
      plotdata1=plotdata1,
      plotdata2=plotdata2,
      lines1=lines1,
      lines2=lines2,
  )
  # Call update function when slider value is changed.
  timeslider.on_changed(update)
  fig.canvas.draw()
  plt.show()
  fig.tight_layout()


def _update(
    newtime,
    plotdata1: PlotData,
    lines1: Sequence[matplotlib.lines.Line2D],
    plotdata2: PlotData | None = None,
    lines2: Sequence[matplotlib.lines.Line2D] | None = None,
):
  """Update plots with new values following slider manipulation."""
  idx = np.abs(plotdata1.t - newtime).argmin()  # find index closest to new time
  # pytype: disable=attribute-error
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
  if plotdata2 is not None and lines2 is not None:
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
    # pytype: enable=attribute-error


def create_slider(
    plotdata1: PlotData,
    plotdata2: PlotData | None = None,
) -> widgets.Slider:
  """Create a slider tool for the plot."""
  plt.subplots_adjust(bottom=0.2)
  axslide = plt.axes([0.12, 0.05, 0.75, 0.05])

  # pytype: disable=attribute-error
  if plotdata2 is not None:
    dt = min(plotdata1.dt, plotdata2.dt)
  else:
    dt = plotdata1.dt

  return widgets.Slider(
      axslide,
      'Time [s]',
      plotdata1.tmin,
      plotdata1.tmax,
      valinit=plotdata1.tmin,
      valstep=dt,
  )


def format_plots(plotdata: PlotData, subfigures: tuple[Any, ...]):
  """Sets up plot formatting."""
  ax1, ax2, ax3, ax4, ax5, ax6 = subfigures

# ax1.set_ylim([0, np.max([plotdata1.ymax_chi_i, plotdata1.ymax_chi_e]) * 1.05])
  # pytype: disable=attribute-error
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
  ax4.legend(fontsize=10)

  ax5.set_ylim([0, plotdata.ymax_q * 1.05])
  ax5.set_xlabel('Normalized radius')
  ax5.set_ylabel('Safety factor')
  ax5.legend()

  ax6.set_ylim([min(plotdata.ymin_s * 1.05, 0), plotdata.ymax_s * 1.05])
  ax6.set_xlabel('Normalized radius')
  ax6.set_ylabel('Magnetic shear')
  ax6.legend()
  # pytype: enable=attribute-error


def get_lines(
    plotdata: PlotData,
    subfigures: tuple[Any, ...],
    comp_plot: bool = False,
):
  """Gets lines for all plots."""
  lines = []
  # If comparison, first lines labeled (1) and solid, second set (2) and dashed.
  if not comp_plot:
    suffix = '~(1)'
    dashed = ''
  else:
    suffix = '~(2)'
    dashed = '--'

  ax1, ax2, ax3, ax4, ax5, ax6 = subfigures

  (line,) = ax1.plot(
      plotdata.rface_coord,
      plotdata.chi_i[1, :],
      'r' + dashed,
      label=rf'$\chi_i{suffix}$',
  )
  lines.append(line)
  (line,) = ax1.plot(
      plotdata.rface_coord,
      plotdata.chi_e[1, :],
      'b' + dashed,
      label=rf'$\chi_e{suffix}$',
  )
  lines.append(line)
  (line,) = ax2.plot(
      plotdata.rcell_coord,
      plotdata.ti[0, :],
      'r' + dashed,
      label=rf'$T_i{suffix}$',
  )
  lines.append(line)
  (line,) = ax2.plot(
      plotdata.rcell_coord,
      plotdata.te[0, :],
      'b' + dashed,
      label=rf'$T_e{suffix}$',
  )
  lines.append(line)
  (line,) = ax3.plot(
      plotdata.rcell_coord,
      plotdata.ne[0, :],
      'r' + dashed,
      label=rf'$n_e{suffix}$',
  )
  lines.append(line)

  (line,) = ax4.plot(
      plotdata.rcell_coord,
      plotdata.j[0, :],
      'r' + dashed,
      label=rf'$j_{{tot}}{suffix}$',
  )
  lines.append(line)
  (line,) = ax4.plot(
      plotdata.rcell_coord,
      plotdata.johm[0, :],
      'b' + dashed,
      label=rf'$j_{{ohm}}{suffix}$',
  )
  lines.append(line)
  (line,) = ax4.plot(
      plotdata.rcell_coord,
      plotdata.j_bootstrap[0, :],
      'g' + dashed,
      label=rf'$j_{{bs}}{suffix}$',
  )
  lines.append(line)
  (line,) = ax4.plot(
      plotdata.rcell_coord,
      plotdata.jext[0, :],
      'm' + dashed,
      label=rf'$j_{{ext}}{suffix}$',
  )
  lines.append(line)
  (line,) = ax5.plot(
      plotdata.rface_coord,
      plotdata.q[0, :],
      'r' + dashed,
      label=rf'$q{suffix}$',
  )
  lines.append(line)
  (line,) = ax6.plot(
      plotdata.rface_coord,
      plotdata.s[0, :],
      'r' + dashed,
      label=rf'$\hat{{s}}{suffix}$',
  )
  lines.append(line)

  return lines


def load_data(filename: str) -> PlotData:
  ds = xr.open_dataset(filename)
  if 'time' in ds:
    t = ds['time'].to_numpy()
  else:
    t = ds['t'].to_numpy()
  return PlotData(
      ti=ds['temp_ion'].to_numpy(),
      te=ds['temp_el'].to_numpy(),
      ne=ds['ne'].to_numpy(),
      j=ds['jtot'].to_numpy(),
      johm=ds['johm'].to_numpy(),
      j_bootstrap=ds['j_bootstrap'].to_numpy(),
      jext=ds['jext'].to_numpy(),
      q=ds['q_face'].to_numpy(),
      s=ds['s_face'].to_numpy(),
      chi_i=ds['chi_face_ion'].to_numpy(),
      chi_e=ds['chi_face_el'].to_numpy(),
      rcell_coord=ds['r_cell_norm'].to_numpy(),
      rface_coord=ds['r_face_norm'].to_numpy(),
      t=t,
  )


def create_figure():
  fig = plt.figure(figsize=(15, 10))
  ax1 = fig.add_subplot(231)
  ax2 = fig.add_subplot(232)
  ax3 = fig.add_subplot(233)
  ax4 = fig.add_subplot(234)
  ax5 = fig.add_subplot(235)
  ax6 = fig.add_subplot(236)
  subfigures = (ax1, ax2, ax3, ax4, ax5, ax6)
  return fig, subfigures
