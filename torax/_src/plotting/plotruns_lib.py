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
from typing import Any, List

import matplotlib
from matplotlib import figure
from matplotlib import gridspec
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
from torax._src.output_tools import output
import xarray as xr

# Internal import.

# Constants for figure setup, plot labels, and formatting.
# The axes are designed to be plotted in the order they appear in the list,
# first ascending in columns, then rows.


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
  r"""Class for all plot related data with support for dynamic variable access.

  This class provides access to both hardcoded variables (for backward compatibility)
  and any variable available in the output file through dynamic attribute access.

  Dynamic Attributes:
    Any variable available in the output file can be accessed by name. The class
    will automatically search through the profiles, scalars, numerics, and
    top-level datasets to find the requested variable. This allows users to plot
    any variable from the output file without needing to add it to the hardcoded
    list of attributes.
  """

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

  def __init__(
      self,
      profiles_dataset: xr.Dataset,
      scalars_dataset: xr.Dataset,
      dataset: xr.Dataset,
      numerics_dataset: xr.Dataset | None = None,
  ):
    """Initialize PlotData with datasets and hardcoded attributes.

    Args:
      profiles_dataset: Dataset containing profile variables.
      scalars_dataset: Dataset containing scalar variables.
      dataset: Top-level dataset containing coordinates.
      numerics_dataset: Dataset containing numeric variables (optional).
    """
    self._profiles_dataset = profiles_dataset
    self._scalars_dataset = scalars_dataset
    self._dataset = dataset
    self._numerics_dataset = numerics_dataset


  def __getattr__(self, name: str) -> np.ndarray:
    """Dynamically access variables from the output datasets.

    This method allows access to any variable in the output file, not just
    the hardcoded ones. It searches through profiles, scalars, numerics,
    and top-level datasets to find the requested variable.

    Args:
      name: Name of the variable to access.

    Returns:
      numpy array containing the variable data.

    Raises:
      AttributeError: If the variable is not found in any dataset.
    """
    # Search in profiles dataset
    if self._profiles_dataset is not None and name in self._profiles_dataset:
      return self._profiles_dataset[name].to_numpy()

    # Search in scalars dataset
    if self._scalars_dataset is not None and name in self._scalars_dataset:
      return self._scalars_dataset[name].to_numpy()

    # Search in top-level dataset (for coordinates and other top-level vars)
    if self._dataset is not None and name in self._dataset:
      return self._dataset[name].to_numpy()

    # Check numerics dataset if it exists
    if self._numerics_dataset is not None and name in self._numerics_dataset:
      return self._numerics_dataset[name].to_numpy()

    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'. "
        f"Variable '{name}' not found in output file datasets."
    )

def flatten_plotdata(plotdata: PlotData) -> dict:
  """Flatten PlotData attributes into a dict for validation.

  Returns all available attributes including properties and dynamic variables.

  Args:
    plotdata: PlotData instance to flatten.

  Returns:
    Dictionary of all available attribute names.
  """
  attrs = {}

  # Add all properties
  for name in dir(type(plotdata)):
    attr = getattr(type(plotdata), name, None)
    if isinstance(attr, property):
      attrs[name] = True

  # Add all dataset variables
  if plotdata._profiles_dataset is not None:
    for var in plotdata._profiles_dataset.data_vars:
      attrs[var] = True

  if plotdata._scalars_dataset is not None:
    for var in plotdata._scalars_dataset.data_vars:
      attrs[var] = True

  if plotdata._dataset is not None:
    for var in plotdata._dataset.data_vars:
      attrs[var] = True

  if plotdata._numerics_dataset is not None:
    for var in plotdata._numerics_dataset.data_vars:
      attrs[var] = True

  return attrs


def load_data(filename: str) -> PlotData:
  """Loads an xr.Dataset from a file, handling potential coordinate name changes."""
  data_tree = output.load_state_file(filename)
  # Handle potential time coordinate name variations
  time = data_tree[output.TIME].to_numpy()

  def get_optional_data(ds, key, grid_type):
    if grid_type.lower() not in ['cell', 'face']:
      raise ValueError(
          f'grid_type for {key} must be either "cell" or "face", got'
          f' {grid_type}'
      )
    if key in ds:
      return ds[key].to_numpy()
    else:
      return (
          np.zeros((len(time), len(ds[output.RHO_CELL_NORM])))
          if grid_type == 'cell'
          else np.zeros((len(time), len(ds[output.RHO_FACE_NORM].to_numpy())))
      )

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

  data_tree = xr.map_over_datasets(_transform_data, data_tree)
  profiles_dataset = data_tree.children[output.PROFILES].dataset
  scalars_dataset = data_tree.children[output.SCALARS].dataset
  dataset = data_tree.dataset
  numerics_dataset = (
      data_tree.children[output.NUMERICS].dataset
      if output.NUMERICS in data_tree.children
      else None
  )

  return PlotData(
      profiles_dataset=profiles_dataset,
      scalars_dataset=scalars_dataset,
      dataset=dataset,
      numerics_dataset=numerics_dataset,
      T_i=profiles_dataset[output.T_I].to_numpy(),
      T_e=profiles_dataset[output.T_E].to_numpy(),
      n_e=profiles_dataset[output.N_E].to_numpy(),
      n_i=profiles_dataset[output.N_I].to_numpy(),
      n_impurity=profiles_dataset[output.N_IMPURITY].to_numpy(),
      Z_impurity=profiles_dataset[output.Z_IMPURITY].to_numpy(),
      psi=profiles_dataset[output.PSI].to_numpy(),
      v_loop=profiles_dataset[output.V_LOOP].to_numpy(),
      j_total=profiles_dataset[output.J_TOROIDAL_TOTAL].to_numpy(),
      j_ohmic=profiles_dataset[output.J_TOROIDAL_OHMIC].to_numpy(),
      j_bootstrap=profiles_dataset[output.J_TOROIDAL_BOOTSTRAP].to_numpy(),
      j_external=profiles_dataset[output.J_TOROIDAL_EXTERNAL].to_numpy(),
      j_ecrh=get_optional_data(profiles_dataset, 'j_ecrh', 'cell'),
      j_generic_current=get_optional_data(
          profiles_dataset, 'j_generic_current', 'cell'
      ),
      q=profiles_dataset[output.Q].to_numpy(),
      magnetic_shear=profiles_dataset[output.MAGNETIC_SHEAR].to_numpy(),
      chi_turb_i=profiles_dataset[output.CHI_TURB_I].to_numpy(),
      chi_neo_i=profiles_dataset[output.CHI_NEO_I].to_numpy(),
      chi_turb_e=profiles_dataset[output.CHI_TURB_E].to_numpy(),
      chi_neo_e=profiles_dataset[output.CHI_NEO_E].to_numpy(),
      D_turb_e=profiles_dataset[output.D_TURB_E].to_numpy(),
      D_neo_e=profiles_dataset[output.D_NEO_E].to_numpy(),
      V_turb_e=profiles_dataset[output.V_TURB_E].to_numpy(),
      V_neo_e=profiles_dataset[output.V_NEO_E].to_numpy(),
      V_neo_ware_e=profiles_dataset[output.V_NEO_WARE_E].to_numpy(),
      rho_norm=dataset[output.RHO_NORM].to_numpy(),
      rho_cell_norm=dataset[output.RHO_CELL_NORM].to_numpy(),
      rho_face_norm=dataset[output.RHO_FACE_NORM].to_numpy(),
      p_icrh_i=get_optional_data(profiles_dataset, 'p_icrh_i', 'cell'),
      p_icrh_e=get_optional_data(profiles_dataset, 'p_icrh_e', 'cell'),
      p_generic_heat_i=get_optional_data(
          profiles_dataset, 'p_generic_heat_i', 'cell'
      ),
      p_generic_heat_e=get_optional_data(
          profiles_dataset, 'p_generic_heat_e', 'cell'
      ),
      p_ecrh_e=get_optional_data(profiles_dataset, 'p_ecrh_e', 'cell'),
      p_alpha_i=get_optional_data(profiles_dataset, 'p_alpha_i', 'cell'),
      p_alpha_e=get_optional_data(profiles_dataset, 'p_alpha_e', 'cell'),
      p_ohmic_e=get_optional_data(profiles_dataset, 'p_ohmic_e', 'cell'),
      p_bremsstrahlung_e=get_optional_data(
          profiles_dataset, 'p_bremsstrahlung_e', 'cell'
      ),
      p_cyclotron_radiation_e=get_optional_data(
          profiles_dataset, 'p_cyclotron_radiation_e', 'cell'
      ),
      p_impurity_radiation_e=get_optional_data(
          profiles_dataset, 'p_impurity_radiation_e', 'cell'
      ),
      ei_exchange=profiles_dataset[
          'ei_exchange'
      ].to_numpy(),  # ion heating/sink
      Q_fusion=scalars_dataset['Q_fusion'].to_numpy(),  # pylint: disable=invalid-name
      s_gas_puff=get_optional_data(profiles_dataset, 's_gas_puff', 'cell'),
      s_generic_particle=get_optional_data(
          profiles_dataset, 's_generic_particle', 'cell'
      ),
      s_pellet=get_optional_data(profiles_dataset, 's_pellet', 'cell'),
      Ip_profile=profiles_dataset[output.IP_PROFILE].to_numpy()[:, -1],
      I_bootstrap=scalars_dataset[output.I_BOOTSTRAP].to_numpy(),
      I_aux_generic=scalars_dataset['I_aux_generic'].to_numpy(),
      I_ecrh=scalars_dataset['I_ecrh'].to_numpy(),
      P_ohmic_e=scalars_dataset['P_ohmic_e'].to_numpy(),
      P_auxiliary=scalars_dataset['P_aux_total'].to_numpy(),
      P_alpha_total=scalars_dataset['P_alpha_total'].to_numpy(),
      P_sink=scalars_dataset['P_bremsstrahlung_e'].to_numpy()
      + scalars_dataset['P_radiation_e'].to_numpy()
      + scalars_dataset['P_cyclotron_e'].to_numpy(),
      P_bremsstrahlung_e=scalars_dataset['P_bremsstrahlung_e'].to_numpy(),
      P_radiation_e=scalars_dataset['P_radiation_e'].to_numpy(),
      P_cyclotron_e=scalars_dataset['P_cyclotron_e'].to_numpy(),
      T_e_volume_avg=scalars_dataset['T_e_volume_avg'].to_numpy(),
      T_i_volume_avg=scalars_dataset['T_i_volume_avg'].to_numpy(),
      n_e_volume_avg=scalars_dataset['n_e_volume_avg'].to_numpy(),
      n_i_volume_avg=scalars_dataset['n_i_volume_avg'].to_numpy(),
      W_thermal_total=scalars_dataset['W_thermal_total'].to_numpy(),
      q95=scalars_dataset['q95'].to_numpy(),
      t=time,
  )


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

  # Attribute check. Sufficient to check one PlotData object.
  # Get available attributes (hardcoded + dynamic)
  if hasattr(plotdata1, '__dataclass_fields__'):
    # Handle both method and dict-like access for backward compatibility
    fields = plotdata1.__dataclass_fields__()
    if callable(fields):
      plotdata_fields = set(fields.keys())
    else:
      plotdata_fields = set(fields.keys() if hasattr(fields, 'keys') else fields)
  else:
    plotdata_fields = set()

  plotdata_properties = {
      name
      for name, _ in inspect.getmembers(
          type(plotdata1), lambda o: isinstance(o, property)
      )
  }
  plotdata_attrs = plotdata_fields.union(plotdata_properties)

  # Validate attributes - try dynamic access for any not in hardcoded list
  for cfg in plot_config.axes:
    for attr in cfg.attrs:
      if attr not in plotdata_attrs:
        # Try to access the attribute dynamically
        try:
          _ = getattr(plotdata1, attr)
          # If successful, add it to the set for future checks
          plotdata_attrs.add(attr)
        except AttributeError:
          raise ValueError(
              f"Attribute '{attr}' in plot_config does not exist in PlotData "
              f"and was not found in the output file datasets."
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
