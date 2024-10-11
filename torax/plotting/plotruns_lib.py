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
from os import path
from typing import Any

import matplotlib
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
from torax import output
import xarray as xr

# Constants for figure setup, plot labels, and formatting.
# The axes are designed to be plotted in the order they appear in the list,
# first ascending in columns, then rows.


@dataclasses.dataclass
class PlotProperties:
  """Dataclass for individual plot properties."""

  attrs: tuple[str, ...]
  labels: tuple[str, ...]
  ylabel: str
  legend_fontsize: int | None = None  # None reverts to default matplotlib value
  upper_percentile: float = 100.0
  lower_percentile: float = 0.0
  include_first_timepoint: bool = True
  ylim_min_zero: bool = True


@dataclasses.dataclass
class FigureProperties:
  """Dataclass for all figure related data."""

  rows: int
  cols: int
  axes: tuple[PlotProperties, ...]
  figure_size_factor: float = 5
  default_legend_fontsize: int = 10
  colors: tuple[str, ...] = ('r', 'b', 'g', 'm', 'y', 'c')

  def __post_init__(self):
    if len(self.axes) > self.rows * self.cols:
      raise ValueError('len(axes) in plot_config is more than rows * columns.')


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
  rho_cell_coord: np.ndarray
  rho_face_coord: np.ndarray


def load_data(filename: str) -> PlotData:
  """Loads an xr.Dataset from a file, handling potential coordinate name changes."""
  ds = xr.open_dataset(filename)
  # Handle potential time coordinate name variations
  t = ds['time'].to_numpy() if 'time' in ds else ds['t'].to_numpy()
  # Rename coordinates if they exist, ensuring compatibility with older datasets
  if 'r_cell' in ds:
    ds = ds.rename({
        'r_cell': 'rho_cell',
        'r_face': 'rho_face',
        'r_cell_norm': 'rho_cell_norm',
        'r_face_norm': 'rho_face_norm',
    })
  # Handle potential jext coordinate name variations
  if output.CORE_PROFILES_JEXT in ds:
    jext = ds[output.CORE_PROFILES_JEXT].to_numpy()
  else:
    jext = ds['jext'].to_numpy()
  return PlotData(
      ti=ds[output.TEMP_ION].to_numpy(),
      te=ds[output.TEMP_EL].to_numpy(),
      ne=ds[output.NE].to_numpy(),
      j=ds[output.JTOT].to_numpy(),
      johm=ds[output.JOHM].to_numpy(),
      j_bootstrap=ds[output.J_BOOTSTRAP].to_numpy(),
      jext=jext,
      q=ds[output.Q_FACE].to_numpy(),
      s=ds[output.S_FACE].to_numpy(),
      chi_i=ds[output.CHI_FACE_ION].to_numpy(),
      chi_e=ds[output.CHI_FACE_EL].to_numpy(),
      rho_cell_coord=ds[output.RHO_CELL_NORM].to_numpy(),
      rho_face_coord=ds[output.RHO_FACE_NORM].to_numpy(),
      t=t,
  )


def plot_run(
    plot_config: FigureProperties, outfile: str, outfile2: str | None = None
):
  """Plots a single run or comparison of two runs."""
  if not path.exists(outfile):
    raise ValueError(f'File {outfile} does not exist.')
  if outfile2 is not None and not path.exists(outfile2):
    raise ValueError(f'File {outfile2} does not exist.')
  plotdata1 = load_data(outfile)
  plotdata2 = load_data(outfile2) if outfile2 else None

  # Attribute check. Sufficient to check one PlotData object.
  plotdata_attrs = set(
      plotdata1.__dataclass_fields__
  )  # Get PlotData attributes
  for cfg in plot_config.axes:
    for attr in cfg.attrs:
      if attr not in plotdata_attrs:
        raise ValueError(
            f"Attribute '{attr}' in plot_config does not exist in PlotData"
        )

  fig, axes = create_figure(plot_config)

  # Title handling:
  title_lines = [f'(1)={outfile}']
  if outfile2:
    title_lines.append(f'(2)={outfile2}')
  fig.suptitle('\n'.join(title_lines))

  lines1 = get_lines(plot_config, plotdata1, axes)
  lines2 = (
      get_lines(plot_config, plotdata2, axes, comp_plot=True)
      if plotdata2
      else None
  )

  format_plots(plot_config, plotdata1, plotdata2, axes)
  timeslider = create_slider(plotdata1, plotdata2)
  fig.canvas.draw()

  update = lambda newtime: _update(
      newtime, plot_config, plotdata1, lines1, plotdata2, lines2
  )
  # Call update function when slider value is changed.
  timeslider.on_changed(update)
  fig.canvas.draw()
  plt.show()
  fig.tight_layout()


def _update(
    newtime,
    plot_config: FigureProperties,
    plotdata1: PlotData,
    lines1: Sequence[matplotlib.lines.Line2D],
    plotdata2: PlotData | None = None,
    lines2: Sequence[matplotlib.lines.Line2D] | None = None,
):
  """Update plots with new values following slider manipulation."""

  def update_lines(plotdata, lines):
    idx = np.abs(plotdata.t - newtime).argmin()
    line_idx = 0
    for cfg in plot_config.axes:  # Iterate through axes based on plot_config
      for attr in cfg.attrs:  # Update all lines in current subplot.
        lines[line_idx].set_ydata(getattr(plotdata, attr)[idx, :])
        line_idx += 1

  update_lines(plotdata1, lines1)
  if plotdata2 and lines2:
    update_lines(plotdata2, lines2)


def create_slider(
    plotdata1: PlotData,
    plotdata2: PlotData | None = None,
) -> widgets.Slider:
  """Create a slider tool for the plot."""
  plt.subplots_adjust(bottom=0.2)
  axslide = plt.axes([0.12, 0.05, 0.75, 0.05])

  tmin = (
      min(plotdata1.t)
      if plotdata2 is None
      else min(min(plotdata1.t), min(plotdata2.t))
  )
  tmax = (
      max(plotdata1.t)
      if plotdata2 is None
      else max(max(plotdata1.t), max(plotdata2.t))
  )

  dt = (
      min(np.diff(plotdata1.t))
      if plotdata2 is None
      else min(min(np.diff(plotdata1.t)), min(np.diff(plotdata2.t)))
  )

  return widgets.Slider(
      axslide,
      'Time [s]',
      tmin,
      tmax,
      valinit=tmin,
      valstep=dt,
  )


def format_plots(
    plot_config: FigureProperties,
    plotdata1: PlotData,
    plotdata2: PlotData | None,
    axes: tuple[Any, ...],
):
  """Sets up plot formatting."""

  # Set default legend fontsize for legends
  matplotlib.rc('legend', fontsize=plot_config.default_legend_fontsize)

  def get_limit(plotdata, attrs, percentile, include_first_timepoint):
    """Gets the limit for a set of attributes based a histogram percentile."""
    if include_first_timepoint:
      values = np.concatenate([getattr(plotdata, attr) for attr in attrs])
    else:
      values = np.concatenate(
          [getattr(plotdata, attr)[1:, :] for attr in attrs]
      )
    return np.percentile(values, percentile)

  for ax, cfg in zip(axes, plot_config.axes):
    ax.set_xlabel('Normalized radius')
    ax.set_ylabel(cfg.ylabel)

    # Get limits for y-axis based on percentile values.
    # 0.0 or 100.0 are special cases for simple min/max values.
    ymin = get_limit(
        plotdata1, cfg.attrs, cfg.lower_percentile, cfg.include_first_timepoint
    )
    ymax = get_limit(
        plotdata1, cfg.attrs, cfg.upper_percentile, cfg.include_first_timepoint
    )

    if plotdata2:
      ymin = min(
          ymin,
          get_limit(
              plotdata2,
              cfg.attrs,
              cfg.lower_percentile,
              cfg.include_first_timepoint,
          ),
      )
      ymax = max(
          ymax,
          get_limit(
              plotdata2,
              cfg.attrs,
              cfg.upper_percentile,
              cfg.include_first_timepoint,
          ),
      )

    lower_bound = ymin / 1.05 if ymin > 0 else ymin * 1.05
    if cfg.ylim_min_zero:
      ax.set_ylim([min(lower_bound, 0), ymax * 1.05])
    else:
      ax.set_ylim([lower_bound, ymax * 1.05])

    ax.legend(fontsize=cfg.legend_fontsize)


def get_rho(
    plotdata: PlotData,
    data_attr: str,
) -> np.ndarray:
  """Gets the correct rho coordinate for the data."""
  datalen = len(getattr(plotdata, data_attr)[0, :])
  if datalen == len(plotdata.rho_cell_coord):
    return plotdata.rho_cell_coord
  elif datalen == len(plotdata.rho_face_coord):
    return plotdata.rho_face_coord
  else:
    raise ValueError(
        f'Data {datalen} does not coincide with either the cell or face grids.'
    )


def get_lines(
    plot_config: FigureProperties,
    plotdata: PlotData,
    axes: tuple[Any, ...],
    comp_plot: bool = False,
):
  """Gets lines for all plots."""
  lines = []
  # If comparison, first lines labeled (1) and solid, second set (2) and dashed.
  suffix = f' ({1 if not comp_plot else 2})'
  dashed = '--' if comp_plot else ''

  for ax, cfg in zip(axes, plot_config.axes):
    line_idx = 0  # Reset color selection cycling for each plot.
    for attr, label in zip(cfg.attrs, cfg.labels):
      rho = get_rho(plotdata, attr)
      (line,) = ax.plot(
          rho,
          getattr(plotdata, attr)[0, :],  # Plot data at time zero
          plot_config.colors[line_idx % len(plot_config.colors)] + dashed,
          label=f'{label}{suffix}',
      )
      lines.append(line)
      line_idx += 1

  return lines


def create_figure(plot_config: FigureProperties):
  """Creates the figure and axes."""
  rows = plot_config.rows
  cols = plot_config.cols
  figsize = (
      cols * plot_config.figure_size_factor,
      rows * plot_config.figure_size_factor,
  )
  fig, axes = plt.subplots(rows, cols, figsize=figsize)
  # Flatten axes array if necessary (for consistent indexing)
  if isinstance(
      axes, np.ndarray
  ):  # Check if it's a NumPy array before flattening
    axes = axes.flatten()
  elif rows > 1 or cols > 1:  # This shouldn't happen, but added as a safety net
    raise ValueError(
        f'Axes is not a numpy array, but should be one since rows={rows},'
        f' cols={cols}'
    )
  else:
    axes = [axes]  # Make axes iterable if only one subplot
  return fig, axes
