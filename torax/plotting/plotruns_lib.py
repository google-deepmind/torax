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
from os import path
from typing import Any, List

import matplotlib
from matplotlib import gridspec
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
from torax import output
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


@dataclasses.dataclass
class PlotData:
  r"""Dataclass for all plot related data.

  Attributes:
    ti: Ion temperature profile [:math:`\mathrm{keV}`] on the cell grid.
    te: Electron temperature profile [:math:`\mathrm{keV}`] on the cell grid.
    ne: Electron density profile [:math:`\mathrm{10^{20} m^{-3}}`] on the cell
      grid.
    ni: Main ion density profile [:math:`\mathrm{10^{20} m^{-3}}`] on the cell
      grid. Corresponds to an bundled ion mixture if specified as such in the
      config.
    nimp: Impurity density profile [:math:`\mathrm{10^{20} m^{-3}}`] on the cell
      grid. Corresponds to an bundled ion mixture if specified as such in the
      config.
    zimp: Average charge state of the impurity species [dimensionless] on the
      cell grid.
    psi: Poloidal flux [:math:`\mathrm{Wb}`] on the cell grid.
    psidot: Time derivative of poloidal flux (loop voltage :math:`V_{loop}`) [V]
      on the cell grid.
    j: Total toroidal current density profile [:math:`\mathrm{MA/m^2}`] on the
      cell grid.
    johm: Ohmic current density profile [:math:`\mathrm{MA/m^2}`] on the cell
      grid.
    j_bootstrap: Bootstrap current density profile [:math:`\mathrm{MA/m^2}`] on
      the cell grid.
    j_ecrh: Electron cyclotron current density profile [:math:`\mathrm{MA/m^2}`]
      on the cell grid.
    generic_current_source: Generic external current source density profile
      [:math:`\mathrm{MA/m^2}`] on the cell grid.
    external_current_source: Total externally driven current source density
      profile [:math:`\mathrm{MA/m^2}`] on the cell grid.
    q: Safety factor (q-profile) [dimensionless] on the face grid.
    s: Magnetic shear profile [dimensionless] on the face grid.
    chi_i: Ion heat conductivity [:math:`\mathrm{m^2/s}`] on the face grid.
    chi_e: Electron heat conductivity [:math:`\mathrm{m^2/s}`] on the face grid.
    d_e: Electron particle diffusivity [:math:`\mathrm{m^2/s}`] on the face
      grid.
    v_e: Electron particle convection velocity [:math:`\mathrm{m/s}`] on the
      face grid.
    q_icrh_i: ICRH ion heating power density [:math:`\mathrm{MW/m^3}`].
    q_icrh_e: ICRH electron heating power density [:math:`\mathrm{MW/m^3}`].
    q_gen_i: Generic ion heating power density [:math:`\mathrm{MW/m^³}`].
    q_gen_e: Generic electron heating power density [:math:`\mathrm{MW/m^3}`].
    q_ecrh: Electron cyclotron heating power density [:math:`\mathrm{MW/m^3}`].
    q_alpha_i: Fusion alpha particle heating power density to ion
      [:math:`\mathrm{MW/m^3}`].
    q_alpha_e: Fusion alpha particle heating power density to electrons
      [:math:`\mathrm{MW/m^3}`].
    q_ohmic: Ohmic heating power density [:math:`\mathrm{MW/m^3}`].
    q_brems: Bremsstrahlung radiation power density [:math:`\mathrm{MW/m^3}`].
    q_cycl: Cyclotron radiation power density [:math:`\mathrm{MW/m^3}`].
    q_ei: Ion-electron heat exchange power density [:math:`\mathrm{MW/m^3}`].
      Positive values denote ion heating, and negative values denote ion cooling
      (electron heating).
    q_rad: Impurity radiation power density [:math:`\mathrm{MW/m^3}`].
    Q_fusion: Fusion power gain (dimensionless).
    s_puff: Gas puff particle source density [:math:`\mathrm{10^{20} m^{-3}
      s^{-1}}`].
    s_generic: Generic particle source density [:math:`\mathrm{10^{20} m^{-3}
      s^{-1}}`].
    s_pellet: Pellet particle source density [:math:`\mathrm{10^{20} m^{-3}
      s^{-1}}`].
    i_total: Total plasma current [:math:`\mathrm{MA}`].
    i_bootstrap: Total bootstrap current [:math:`\mathrm{MA}`].
    i_generic: Total generic current source [:math:`\mathrm{MA}`].
    i_ecrh: Total electron cyclotron current [:math:`\mathrm{MA}`].
    p_auxiliary: Total auxiliary heating power [:math:`\mathrm{MW}`].
    p_ohmic: Total Ohmic heating power [:math:`\mathrm{MW}`].
    p_alpha: Total fusion alpha heating power [:math:`\mathrm{MW}`].
    p_sink: Total electron heating sink [:math:`\mathrm{MW}`].
    p_brems: Total bremsstrahlung radiation power loss [:math:`\mathrm{MW}`].
    p_cycl: Total cyclotron radiation power loss [:math:`\mathrm{MW}`].
    p_rad: Total impurity radiation power loss [:math:`\mathrm{MW}`].
    t: Simulation time [:math:`\mathrm{s}`].
    rho_cell_coord: Normalized toroidal flux coordinate on the cell grid.
    rho_face_coord: Normalized toroidal flux coordinate on the face grid.
    te_volume_avg: Volume-averaged electron temperature [:math:`\mathrm{keV}`].
    ti_volume_avg: Volume-averaged ion temperature [:math:`\mathrm{keV}`].
    ne_volume_avg: Volume-averaged electron density [:math:`\mathrm{10^{20}
      m^{-3}}`].
    ni_volume_avg: Volume-averaged ion density [:math:`\mathrm{10^{20}
      m^{-3}}`].
    W_thermal_tot: Total thermal stored energy [:math:`\mathrm{MJ}`].
    q95: Safety factor at 95% of the normalized poloidal flux.
  """

  ti: np.ndarray
  te: np.ndarray
  ne: np.ndarray
  ni: np.ndarray
  nimp: np.ndarray
  zimp: np.ndarray
  psi: np.ndarray
  psidot: np.ndarray
  j: np.ndarray
  johm: np.ndarray
  j_bootstrap: np.ndarray
  j_ecrh: np.ndarray
  generic_current_source: np.ndarray
  external_current_source: np.ndarray
  q: np.ndarray
  s: np.ndarray
  chi_i: np.ndarray
  chi_e: np.ndarray
  d_e: np.ndarray
  v_e: np.ndarray
  q_icrh_i: np.ndarray
  q_icrh_e: np.ndarray
  q_gen_i: np.ndarray
  q_gen_e: np.ndarray
  q_ecrh: np.ndarray
  q_alpha_i: np.ndarray
  q_alpha_e: np.ndarray
  q_ohmic: np.ndarray
  q_brems: np.ndarray
  q_cycl: np.ndarray
  q_ei: np.ndarray
  q_rad: np.ndarray
  Q_fusion: np.ndarray  # pylint: disable=invalid-name
  s_puff: np.ndarray
  s_generic: np.ndarray
  s_pellet: np.ndarray
  i_total: np.ndarray
  i_bootstrap: np.ndarray
  i_generic: np.ndarray
  i_ecrh: np.ndarray
  p_auxiliary: np.ndarray
  p_ohmic: np.ndarray
  p_alpha: np.ndarray
  p_sink: np.ndarray
  p_brems: np.ndarray
  p_cycl: np.ndarray
  p_rad: np.ndarray
  t: np.ndarray
  rho_cell_coord: np.ndarray
  rho_face_coord: np.ndarray
  te_volume_avg: np.ndarray
  ti_volume_avg: np.ndarray
  ne_volume_avg: np.ndarray
  ni_volume_avg: np.ndarray
  W_thermal_tot: np.ndarray  # pylint: disable=invalid-name
  q95: np.ndarray


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
    ds = ds.copy()

    transformations = {
        output.J_TOTAL: 1e6,  # A/m^2 to MA/m^2
        output.J_OHMIC: 1e6,  # A/m^2 to MA/m^2
        output.J_BOOTSTRAP: 1e6,  # A/m^2 to MA/m^2
        output.J_EXTERNAL: 1e6,  # A/m^2 to MA/m^2
        'generic_current_source': 1e6,  # A/m^2 to MA/m^2
        output.I_BOOTSTRAP: 1e6,  # A to MA
        output.IP_PROFILE: 1e6,  # A to MA
        'electron_cyclotron_source_j': 1e6,  # A/m^2 to MA/m^2
        'ion_cyclotron_source_ion': 1e6,  # W/m^3 to MW/m^3
        'ion_cyclotron_source_el': 1e6,  # W/m^3 to MW/m^3
        'nbi_heat_source_ion': 1e6,  # W/m^3 to MW/m^3
        'nbi_heat_source_el': 1e6,  # W/m^3 to MW/m^3
        'generic_ion_el_heat_source_ion': 1e6,  # W/m^3 to MW/m^3
        'generic_ion_el_heat_source_el': 1e6,  # W/m^3 to MW/m^3
        'electron_cyclotron_source_el': 1e6,  # W/m^3 to MW/m^3
        'fusion_heat_source_ion': 1e6,  # W/m^3 to MW/m^3
        'fusion_heat_source_el': 1e6,  # W/m^3 to MW/m^3
        'ohmic_heat_source': 1e6,  # W/m^3 to MW/m^3
        'bremsstrahlung_heat_sink': 1e6,  # W/m^3 to MW/m^3
        'cyclotron_radiation_heat_sink': 1e6,  # W/m^3 to MW/m^3
        'impurity_radiation_heat_sink': 1e6,  # W/m^3 to MW/m^3
        'qei_source': 1e6,  # W/m^3 to MW/m^3
        'P_ohmic': 1e6,  # W to MW
        'P_external_tot': 1e6,  # W to MW
        'P_alpha_tot': 1e6,  # W to MW
        'P_brems': 1e6,  # W to MW
        'P_cycl': 1e6,  # W to MW
        'P_ecrh': 1e6,  # W to MW
        'P_rad': 1e6,  # W to MW
        'I_ecrh': 1e6,  # A to MA
        'I_generic': 1e6,  # A to MA
        'W_thermal_tot': 1e6,  # J to MJ
    }

    for var_name, scale in transformations.items():
      if var_name in ds:
        ds[var_name] /= scale

    # For density transformations we need nref which is only available in the
    # core_profiles dataset. Do these separately.
    if output.N_REF in ds:
      ds[output.N_E] *= ds[output.N_REF][0].values / 1e20
      ds[output.N_I] *= ds[output.N_REF][0].values / 1e20
      ds[output.N_IMPURITY] *= ds[output.N_REF][0].values / 1e20

    return ds

  data_tree = xr.map_over_datasets(_transform_data, data_tree)
  core_profiles_dataset = data_tree.children[output.CORE_PROFILES].dataset
  core_sources_dataset = data_tree.children[output.CORE_SOURCES].dataset
  core_transport_dataset = data_tree.children[output.CORE_TRANSPORT].dataset
  post_processed_outputs_dataset = data_tree.children[
      output.POST_PROCESSED_OUTPUTS
  ].dataset
  dataset = data_tree.dataset

  return PlotData(
      ti=core_profiles_dataset[output.TEMPERATURE_ION].to_numpy(),
      te=core_profiles_dataset[output.TEMPERATURE_ELECTRON].to_numpy(),
      ne=core_profiles_dataset[output.N_E].to_numpy(),
      ni=core_profiles_dataset[output.N_I].to_numpy(),
      nimp=core_profiles_dataset[output.N_IMPURITY].to_numpy(),
      zimp=core_profiles_dataset[output.Z_IMPURITY].to_numpy(),
      psi=core_profiles_dataset[output.PSI].to_numpy(),
      psidot=core_profiles_dataset[output.V_LOOP].to_numpy(),
      j=core_profiles_dataset[output.J_TOTAL].to_numpy(),
      johm=core_profiles_dataset[output.J_OHMIC].to_numpy(),
      j_bootstrap=core_profiles_dataset[output.J_BOOTSTRAP].to_numpy(),
      external_current_source=core_profiles_dataset[
          output.J_EXTERNAL
      ].to_numpy(),
      j_ecrh=get_optional_data(
          core_sources_dataset, 'electron_cyclotron_source_j', 'cell'
      ),
      generic_current_source=get_optional_data(
          core_sources_dataset, 'generic_current_source', 'cell'
      ),
      q=core_profiles_dataset[output.Q].to_numpy(),
      s=core_profiles_dataset[output.MAGNETIC_SHEAR].to_numpy(),
      chi_i=core_transport_dataset[output.CHI_TURB_I].to_numpy(),
      chi_e=core_transport_dataset[output.CHI_TURB_E].to_numpy(),
      d_e=core_transport_dataset[output.D_TURB_E].to_numpy(),
      v_e=core_transport_dataset[output.V_TURB_E].to_numpy(),
      rho_cell_coord=dataset[output.RHO_CELL_NORM].to_numpy(),
      rho_face_coord=dataset[output.RHO_FACE_NORM].to_numpy(),
      q_icrh_i=get_optional_data(
          core_sources_dataset, 'ion_cyclotron_source_ion', 'cell'
      ),
      q_icrh_e=get_optional_data(
          core_sources_dataset, 'ion_cyclotron_source_el', 'cell'
      ),
      q_gen_i=get_optional_data(
          core_sources_dataset, 'generic_ion_el_heat_source_ion', 'cell'
      ),
      q_gen_e=get_optional_data(
          core_sources_dataset, 'generic_ion_el_heat_source_el', 'cell'
      ),
      q_ecrh=get_optional_data(
          core_sources_dataset, 'electron_cyclotron_source_el', 'cell'
      ),
      q_alpha_i=get_optional_data(
          core_sources_dataset, 'fusion_heat_source_ion', 'cell'
      ),
      q_alpha_e=get_optional_data(
          core_sources_dataset, 'fusion_heat_source_el', 'cell'
      ),
      q_ohmic=get_optional_data(
          core_sources_dataset, 'ohmic_heat_source', 'cell'
      ),
      q_brems=get_optional_data(
          core_sources_dataset, 'bremsstrahlung_heat_sink', 'cell'
      ),
      q_cycl=get_optional_data(
          core_sources_dataset, 'cyclotron_radiation_heat_sink', 'cell'
      ),
      q_rad=get_optional_data(
          core_sources_dataset, 'impurity_radiation_heat_sink', 'cell'
      ),
      q_ei=core_sources_dataset['qei_source'].to_numpy(),  # ion heating/sink
      Q_fusion=post_processed_outputs_dataset['Q_fusion'].to_numpy(),  # pylint: disable=invalid-name
      s_puff=get_optional_data(core_sources_dataset, 'gas_puff_source', 'cell'),
      s_generic=get_optional_data(
          core_sources_dataset, 'generic_particle_source', 'cell'
      ),
      s_pellet=get_optional_data(core_sources_dataset, 'pellet_source', 'cell'),
      i_total=core_profiles_dataset[output.IP_PROFILE].to_numpy()[:, -1],
      i_bootstrap=core_profiles_dataset[output.I_BOOTSTRAP].to_numpy(),
      i_generic=post_processed_outputs_dataset['I_generic'].to_numpy(),
      i_ecrh=post_processed_outputs_dataset['I_ecrh'].to_numpy(),
      p_ohmic=post_processed_outputs_dataset['P_ohmic'].to_numpy(),
      p_auxiliary=(
          post_processed_outputs_dataset['P_external_tot']
          - post_processed_outputs_dataset['P_ohmic']
      ).to_numpy(),
      p_alpha=post_processed_outputs_dataset['P_alpha_tot'].to_numpy(),
      p_sink=post_processed_outputs_dataset['P_brems'].to_numpy()
      + post_processed_outputs_dataset['P_rad'].to_numpy()
      + post_processed_outputs_dataset['P_cycl'].to_numpy(),
      p_brems=post_processed_outputs_dataset['P_brems'].to_numpy(),
      p_rad=post_processed_outputs_dataset['P_rad'].to_numpy(),
      p_cycl=post_processed_outputs_dataset['P_cycl'].to_numpy(),
      te_volume_avg=post_processed_outputs_dataset['te_volume_avg'].to_numpy(),
      ti_volume_avg=post_processed_outputs_dataset['ti_volume_avg'].to_numpy(),
      ne_volume_avg=post_processed_outputs_dataset['ne_volume_avg'].to_numpy(),
      ni_volume_avg=post_processed_outputs_dataset['ni_volume_avg'].to_numpy(),
      W_thermal_tot=post_processed_outputs_dataset['W_thermal_tot'].to_numpy(),
      q95=post_processed_outputs_dataset['q95'].to_numpy(),
      t=time,
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

  fig.canvas.draw()
  plt.show()


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
      values = np.concatenate([getattr(plotdata, attr) for attr in attrs])
    else:
      values = np.concatenate(
          [getattr(plotdata, attr)[1:, :] for attr in attrs]
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
