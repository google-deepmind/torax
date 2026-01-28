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
import enum
import inspect
from os import path
from typing import Any, Final, List, Mapping, Set

import immutabledict
import matplotlib
from matplotlib import figure
from matplotlib import gridspec
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
from torax._src.output_tools import output
import xarray as xr

# Internal import.


# For some use-cases it is useful to compare simulations where one will have
# a given source (e.g. p_icrh) and the other does not. However, the simulation
# without the source will not have that variable existing in the profile child
# data tree. To avoid confusion in the plots, it's sometimes useful to
# explicitly display the zero values of the source profile in the simulation
# where that source is off or non-existent. We thus need to return zero profiles
# in the PlotData getattr when that attribute is not initialized from the xarray
# object. The set of profiles for which this is done, is determined by the
# _OPTIONAL_PROFILE_ATTRS dictionary. The key is the variable name. The value
# indicates the appropriate grid type via the _GridType enum.
class _GridType(enum.Enum):
  """Enum for grid types."""

  CELL = enum.auto()
  FACE = enum.auto()


_OPTIONAL_PROFILE_ATTRS: Final[Mapping[str, _GridType]] = (
    immutabledict.immutabledict({
        'j_ecrh': _GridType.CELL,
        'j_generic_current': _GridType.CELL,
        'p_icrh_i': _GridType.CELL,
        'p_icrh_e': _GridType.CELL,
        'p_generic_heat_i': _GridType.CELL,
        'p_generic_heat_e': _GridType.CELL,
        'p_ecrh_e': _GridType.CELL,
        'p_alpha_i': _GridType.CELL,
        'p_alpha_e': _GridType.CELL,
        'p_ohmic_e': _GridType.CELL,
        'p_bremsstrahlung_e': _GridType.CELL,
        'p_cyclotron_radiation_e': _GridType.CELL,
        'p_impurity_radiation_e': _GridType.CELL,
        's_gas_puff': _GridType.CELL,
        's_generic_particle': _GridType.CELL,
        's_pellet': _GridType.CELL,
    })
)


class PlotType(enum.Enum):
  """Enum for plot types.

  SPATIAL: Spatial plots, e.g., 1D profiles as a function of toroidal flux
  coordinate. Plots change with time, and are modified by the slider.
  TIME_SERIES: Time series plots. 0D profiles plotting as a function of time.
  These plots are not modified by the slider.
  """

  SPATIAL = 1
  TIME_SERIES = 2


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
  plot_type: PlotType = PlotType.SPATIAL
  suppress_zero_values: bool = False  # If True, all-zero-data is not plotted


@dataclasses.dataclass
class FigureProperties:
  """Dataclass for all figure related data."""

  rows: int
  cols: int
  axes: tuple[PlotProperties, ...]
  figure_size_factor: float = 5.0
  tick_fontsize: int = 10
  axes_fontsize: int = 10
  title_fontsize: int = 16
  default_legend_fontsize: int = 10
  colors: tuple[str, ...] = ('r', 'b', 'g', 'm', 'y', 'c')

  def __post_init__(self):
    if len(self.axes) > self.rows * self.cols:
      raise ValueError('len(axes) in plot_config is more than rows * columns.')

  @property
  def contains_spatial_plot_type(self) -> bool:
    """Checks if any plot is a spatial plottype."""
    return any(
        plot_properties.plot_type == PlotType.SPATIAL
        for plot_properties in self.axes
    )


# pylint: disable=invalid-name
class PlotData:
  """Class for all plot related data with dynamic variable access.

  All variables from the output file datasets are accessible as attributes.
  """

  def __init__(
      self,
      data_tree: xr.DataTree,
  ):
    """Initialize PlotData with TORAX output DataTree."""
    self._top_level_dataset = data_tree.dataset
    if output.TIME not in self._top_level_dataset:
      raise ValueError('Time variable not found in top-level dataset.')
    self._scalars_dataset = data_tree.children[output.SCALARS].dataset
    self._profiles_dataset = data_tree.children[output.PROFILES].dataset
    self._numerics_dataset = data_tree.children[output.NUMERICS].dataset

  @property
  def chi_total_i(self) -> np.ndarray:
    return self.chi_turb_i + self.chi_neo_i

  @property
  def chi_total_e(self) -> np.ndarray:
    return self.chi_turb_e + self.chi_neo_e

  @property
  def D_total_e(self) -> np.ndarray:
    return self.D_turb_e + self.D_neo_e

  @property
  def V_neo_total_e(self) -> np.ndarray:
    return self.V_neo_e + self.V_neo_ware_e

  @property
  def V_total_e(self) -> np.ndarray:
    return self.V_turb_e + self.V_neo_total_e

  @property
  def P_sink(self) -> np.ndarray:
    """Total electron heating sink power [MW].

    Calculated as sum of bremsstrahlung, radiation, and cyclotron losses.
    """
    return (
        self._scalars_dataset['P_bremsstrahlung_e'].to_numpy()
        + self._scalars_dataset['P_radiation_e'].to_numpy()
        + self._scalars_dataset['P_cyclotron_e'].to_numpy()
    )

  @property
  def P_auxiliary(self) -> np.ndarray:
    """Total auxiliary power [MW]."""
    return self._scalars_dataset['P_aux_total'].to_numpy()

  @property
  def t(self) -> np.ndarray:
    """Accessor for the time coordinate."""
    return self._top_level_dataset[output.TIME].to_numpy()

  def __getattr__(self, name: str) -> np.ndarray:
    """Dynamically access variables from the output datasets.

    Args:
      name: Name of the variable to access.

    Returns:
      A numpy array containing the variable data.
    """

    # Intercept Ip_profile and set as scalars.Ip
    # This is needed for backwards compatibility with V1 Ip_profile definition
    # in PlotData.
    # TODO(b/379838765): Remove this in V2
    if name == 'Ip_profile':
      return self._scalars_dataset['Ip'].to_numpy()

    # 1. Search in profiles dataset
    if self._profiles_dataset is not None and name in self._profiles_dataset:
      return self._profiles_dataset[name].to_numpy()

    # 2. Search in scalars dataset
    if self._scalars_dataset is not None and name in self._scalars_dataset:
      return self._scalars_dataset[name].to_numpy()

    # 3. Search in top-level dataset (for coordinates etc)
    if self._top_level_dataset is not None and name in self._top_level_dataset:
      return self._top_level_dataset[name].to_numpy()

    # 4. Search in numerics dataset
    if self._numerics_dataset is not None and name in self._numerics_dataset:
      return self._numerics_dataset[name].to_numpy()

    # 5. Check if it is a known optional variable that defaults to zero
    if name in _OPTIONAL_PROFILE_ATTRS:
      return self._get_zero_profile(name, _OPTIONAL_PROFILE_ATTRS[name])

    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'. "
        f"Variable '{name}' not found in output file datasets."
    )

  def _get_zero_profile(self, name: str, grid_type: _GridType) -> np.ndarray:
    """Generates a zero-filled array with the correct shape."""
    time_steps = len(self.t)
    match grid_type:
      case _GridType.CELL:
        spatial_steps = len(self.rho_cell_norm)
      case _GridType.FACE:
        spatial_steps = len(self.rho_face_norm)
      case _:
        raise ValueError(f"Unknown grid type '{grid_type}' for {name}")
    return np.zeros((time_steps, spatial_steps))

  def available_variables(self) -> Set[str]:
    """Returns a set of all available attribute names for validation."""
    attrs = set()

    # Add all properties (from the class definition)
    for name, _ in inspect.getmembers(
        type(self), lambda x: isinstance(x, property)
    ):
      attrs.add(name)

    datasets = [
        self._profiles_dataset,
        self._scalars_dataset,
        self._top_level_dataset,
        self._numerics_dataset,
    ]

    for ds in datasets:
      if ds is not None:
        for var in ds.data_vars:
          attrs.add(str(var))

    for name in _OPTIONAL_PROFILE_ATTRS:
      attrs.add(name)

    return attrs


def load_data(filename: str) -> PlotData:
  """Loads an xr.Dataset from a file, handling potential coordinate name changes."""
  data_tree = output.load_state_file(filename)

  def _transform_data(ds: xr.Dataset):
    """Transforms data in-place to the desired units."""
    # TODO(b/414755419)
    ds = ds.copy()

    transformations = {
        output.J_TOROIDAL_TOTAL: 1e6,  # A/m^2 to MA/m^2
        output.J_TOROIDAL_OHMIC: 1e6,  # A/m^2 to MA/m^2
        output.J_TOROIDAL_BOOTSTRAP: 1e6,  # A/m^2 to MA/m^2
        output.J_TOROIDAL_EXTERNAL: 1e6,  # A/m^2 to MA/m^2
        'j_generic_current': 1e6,  # A/m^2 to MA/m^2
        output.I_BOOTSTRAP: 1e6,  # A to MA
        output.IP_PROFILE: 1e6,  # A to MA
        'j_ecrh': 1e6,  # A/m^2 to MA/m^2
        'p_icrh_i': 1e6,  # W/m^3 to MW/m^3
        'p_icrh_e': 1e6,  # W/m^3 to MW/m^3
        'p_generic_heat_i': 1e6,  # W/m^3 to MW/m^3
        'p_generic_heat_e': 1e6,  # W/m^3 to MW/m^3
        'p_ecrh_e': 1e6,  # W/m^3 to MW/m^3
        'p_alpha_i': 1e6,  # W/m^3 to MW/m^3
        'p_alpha_e': 1e6,  # W/m^3 to MW/m^3
        'p_ohmic_e': 1e6,  # W/m^3 to MW/m^3
        'p_bremsstrahlung_e': 1e6,  # W/m^3 to MW/m^3
        'p_cyclotron_radiation_e': 1e6,  # W/m^3 to MW/m^3
        'p_impurity_radiation_e': 1e6,  # W/m^3 to MW/m^3
        'ei_exchange': 1e6,  # W/m^3 to MW/m^3
        'P_ohmic_e': 1e6,  # W to MW
        'P_aux_total': 1e6,  # W to MW
        'P_alpha_total': 1e6,  # W to MW
        'P_bremsstrahlung_e': 1e6,  # W to MW
        'P_cyclotron_e': 1e6,  # W to MW
        'P_ecrh': 1e6,  # W to MW
        'P_radiation_e': 1e6,  # W to MW
        'I_ecrh': 1e6,  # A to MA
        'I_aux_generic': 1e6,  # A to MA
        'W_thermal_total': 1e6,  # J to MJ
        output.N_E: 1e20,  # m^-3 to 10^{20} m^-3
        output.N_I: 1e20,  # m^-3 to 10^{20} m^-3
        output.N_IMPURITY: 1e20,  # m^-3 to 10^{20} m^-3
    }

    for var_name, scale in transformations.items():
      if var_name in ds:
        ds[var_name] /= scale

    return ds

  return PlotData(xr.map_over_datasets(_transform_data, data_tree))


def plot_run(
    plot_config: FigureProperties,
    outfile: str,
    outfile2: str | None = None,
    interactive: bool = True,
) -> figure.Figure:
  """Plots a single run or comparison of two runs."""
  if not path.exists(outfile):
    raise ValueError(f'File {outfile} does not exist.')
  if outfile2 is not None and not path.exists(outfile2):
    raise ValueError(f'File {outfile2} does not exist.')
  plotdata1 = load_data(outfile)
  plotdata2 = load_data(outfile2) if outfile2 else None

  # Prepare list of datasets to check, associating them with their filenames
  # for clearer errors
  datasets_to_check = [(plotdata1, outfile)]
  if plotdata2 is not None:
    datasets_to_check.append((plotdata2, outfile2))

  for plotdata, filename in datasets_to_check:
    # Get the set of valid keys for this specific dataset
    available_vars = plotdata.available_variables()

    for cfg in plot_config.axes:
      for attr in cfg.attrs:
        if attr not in available_vars:
          raise ValueError(
              f"Attribute '{attr}' in plot_config was not found in "
              f'output file: {filename}'
          )

  fig, axes, slider_ax = create_figure(plot_config)

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

  # Only create the slider if needed.
  if plot_config.contains_spatial_plot_type:
    timeslider = create_slider(slider_ax, plotdata1, plotdata2)

    def update(newtime):
      """Update plots with new values following slider manipulation."""
      fig.constrained_layout = False
      _update(newtime, plot_config, plotdata1, lines1, plotdata2, lines2)
      fig.constrained_layout = True
      fig.canvas.draw_idle()

    timeslider.on_changed(update)

  if interactive:
    fig.canvas.draw()
    plt.show()
  return fig


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
      if cfg.plot_type == PlotType.TIME_SERIES:
        continue  # Time series plots do not need to be updated
      for attr in cfg.attrs:  # Update all lines in current subplot.
        data = getattr(plotdata, attr)
        if cfg.suppress_zero_values and np.all(data == 0):
          continue
        lines[line_idx].set_ydata(data[idx, :])
        line_idx += 1

  update_lines(plotdata1, lines1)
  if plotdata2 and lines2:
    update_lines(plotdata2, lines2)


def create_slider(
    ax: matplotlib.axes.Axes,
    plotdata1: PlotData,
    plotdata2: PlotData | None = None,
) -> widgets.Slider:
  """Create a slider tool for the plot."""
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
      ax,
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
    axes: List[Any],
):
  """Sets up plot formatting."""

  # Set default legend fontsize for legends
  matplotlib.rc('legend', fontsize=plot_config.default_legend_fontsize)

  def get_limit(plotdata, attrs, percentile, include_first_timepoint):
    """Gets the limit for a set of attributes based a histogram percentile."""
    if include_first_timepoint:
      values = np.concatenate(
          [getattr(plotdata, attr).flatten() for attr in attrs]
      )
    else:
      values = np.concatenate(
          [getattr(plotdata, attr)[1:, :].flatten() for attr in attrs]
      )
    return np.percentile(values, percentile)

  for ax, cfg in zip(axes, plot_config.axes):
    if cfg.plot_type == PlotType.SPATIAL:
      ax.set_xlabel('Normalized radius')
    elif cfg.plot_type == PlotType.TIME_SERIES:
      ax.set_xlabel('Time [s]')
    else:
      raise ValueError(f'Unknown plot type: {cfg.plot_type}')
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

    # Guard against empty data
    if ymax != 0 or ymin != 0:  # Check for meaningful data range
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
  if datalen == len(plotdata.rho_cell_norm):
    return plotdata.rho_cell_norm
  elif datalen == len(plotdata.rho_face_norm):
    return plotdata.rho_face_norm
  elif datalen == len(plotdata.rho_norm):
    return plotdata.rho_norm
  else:
    raise ValueError(
        f'Data {datalen} does not coincide with either the cell or face grids.'
    )


def get_lines(
    plot_config: FigureProperties,
    plotdata: PlotData,
    axes: List[Any],
    comp_plot: bool = False,
):
  """Gets lines for all plots."""
  lines = []
  # If comparison, first lines labeled (1) and solid, second set (2) and dashed.
  suffix = f' ({1 if not comp_plot else 2})'
  dashed = '--' if comp_plot else ''

  for ax, cfg in zip(axes, plot_config.axes):
    line_idx = 0  # Reset color selection cycling for each plot.
    if cfg.plot_type == PlotType.SPATIAL:
      for attr, label in zip(cfg.attrs, cfg.labels):
        data = getattr(plotdata, attr)
        if cfg.suppress_zero_values and np.all(data == 0):
          continue
        rho = get_rho(plotdata, attr)
        (line,) = ax.plot(
            rho,
            data[0, :],  # Plot data at time zero
            plot_config.colors[line_idx % len(plot_config.colors)] + dashed,
            label=f'{label}{suffix}',
        )
        lines.append(line)
        line_idx += 1
    elif cfg.plot_type == PlotType.TIME_SERIES:
      for attr, label in zip(cfg.attrs, cfg.labels):
        data = getattr(plotdata, attr)
        if cfg.suppress_zero_values and np.all(data == 0):
          continue
        # No need to return a line since this will not need to be updated.
        _ = ax.plot(
            plotdata.t,
            data,  # Plot entire time series
            plot_config.colors[line_idx % len(plot_config.colors)] + dashed,
            label=f'{label}{suffix}',
        )
        line_idx += 1
    else:
      raise ValueError(f'Unknown plot type: {cfg.plot_type}')

  return lines


def create_figure(plot_config: FigureProperties):
  """Creates the figure and axes."""
  rows = plot_config.rows
  cols = plot_config.cols
  matplotlib.rc('xtick', labelsize=plot_config.tick_fontsize)
  matplotlib.rc('ytick', labelsize=plot_config.tick_fontsize)
  matplotlib.rc('axes', labelsize=plot_config.axes_fontsize)
  matplotlib.rc('figure', titlesize=plot_config.title_fontsize)
  fig = plt.figure(
      figsize=(
          cols * plot_config.figure_size_factor,
          rows * plot_config.figure_size_factor,
      ),
      constrained_layout=True,
  )
  # Create the GridSpec - Adjust height ratios to include the slider
  # in the plot, only if a slider is required:
  if plot_config.contains_spatial_plot_type:
    # Add an extra smaller is a spatial plottypeider
    height_ratios = [1] * rows + [0.2]
    gs = gridspec.GridSpec(
        rows + 1, cols, figure=fig, height_ratios=height_ratios
    )
    # slider spans all columns
    slider_ax = fig.add_subplot(gs[rows, :])
  else:
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    slider_ax = None

  axes = []
  for i in range(rows * cols):
    row = i // cols
    col = i % cols
    axes.append(fig.add_subplot(gs[row, col]))  # Add subplots to the grid
  return fig, axes, slider_ax
