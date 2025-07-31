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
@dataclasses.dataclass
class PlotData:
  r"""Dataclass for all plot related data.

  Attributes:
    T_i: Ion temperature profile [:math:`\mathrm{keV}`] on the cell grid.
    T_e: Electron temperature profile [:math:`\mathrm{keV}`] on the cell grid.
    n_e: Electron density profile [:math:`\mathrm{10^{20} m^{-3}}`] on the cell
      grid.
    n_i: Main ion density profile [:math:`\mathrm{10^{20} m^{-3}}`] on the cell
      grid. Corresponds to a bundled ion mixture if specified as such in the
      config.
    n_impurity: Impurity density profile [:math:`\mathrm{10^{20} m^{-3}}`] on
      the cell grid. Corresponds to a bundled ion mixture if specified as such
      in the config.
    Z_impurity: Average charge state of the impurity species [dimensionless] on
      the cell grid.
    psi: Poloidal flux [:math:`\mathrm{Wb}`] on the cell grid.
    v_loop: Time derivative of poloidal flux (loop voltage :math:`V_{loop}`) [V]
      on the cell grid.
    j_total: Total toroidal current density profile [:math:`\mathrm{MA/m^2}`] on
      the cell grid.
    j_ohmic: Ohmic current density profile [:math:`\mathrm{MA/m^2}`] on the cell
      grid.
    j_bootstrap: Bootstrap current density profile [:math:`\mathrm{MA/m^2}`] on
      the cell grid.
    j_ecrh: Electron cyclotron current density profile [:math:`\mathrm{MA/m^2}`]
      on the cell grid.
    j_generic_current: Generic external current source density profile
      [:math:`\mathrm{MA/m^2}`] on the cell grid.
    j_external: Total externally driven current source density profile
      [:math:`\mathrm{MA/m^2}`] on the cell grid.
    q: Safety factor (q-profile) [dimensionless] on the face grid.
    magnetic_shear: Magnetic shear profile [dimensionless] on the face grid.
    chi_turb_i: Turbulent ion heat conductivity [:math:`\mathrm{m^2/s}`] on the
      face grid.
    chi_neo_i: Neoclassical ion heat conductivity [:math:`\mathrm{m^2/s}`] on
      the face grid.
    chi_turb_e: Turbulent electron heat conductivity [:math:`\mathrm{m^2/s}`] on
      the face grid.
    chi_neo_e: Neoclassical electron heat conductivity [:math:`\mathrm{m^2/s}`]
      on the face grid.
    D_turb_e: Turbulent electron particle diffusivity [:math:`\mathrm{m^2/s}`]
      on the face grid.
    D_neo_e: Neoclassical electron particle diffusivity [:math:`\mathrm{m^2/s}`]
      on the face grid.
    V_turb_e: Turbulent electron particle convection [:math:`\mathrm{m^2/s}`] on
      the face grid.
    V_neo_e: Neoclassical electron particle convection [:math:`\mathrm{m^2/s}`]
      on the face grid. Contains all components apart from the Ware pinch.
    V_neo_ware_e: Neoclassical electron particle convection (Ware pinch term)
      [:math:`\mathrm{m^2/s}`] on the face grid.
    p_icrh_i: ICRH ion heating power density [:math:`\mathrm{MW/m^3}`].
    p_icrh_e: ICRH electron heating power density [:math:`\mathrm{MW/m^3}`].
    p_generic_heat_i: Generic ion heating power density
      [:math:`\mathrm{MW/m^³}`].
    p_generic_heat_e: Generic electron heating power density
      [:math:`\mathrm{MW/m^3}`].
    p_ecrh_e: Electron cyclotron heating power density
      [:math:`\mathrm{MW/m^3}`].
    p_alpha_i: Fusion alpha particle heating power density to ion
      [:math:`\mathrm{MW/m^3}`].
    p_alpha_e: Fusion alpha particle heating power density to electrons
      [:math:`\mathrm{MW/m^3}`].
    p_ohmic_e: Ohmic heating power density [:math:`\mathrm{MW/m^3}`].
    p_bremsstrahlung_e: Bremsstrahlung radiation power density
      [:math:`\mathrm{MW/m^3}`].
    p_cyclotron_radiation_e: Cyclotron radiation power density
      [:math:`\mathrm{MW/m^3}`].
    ei_exchange: Ion-electron heat exchange power density
      [:math:`\mathrm{MW/m^3}`]. Positive values denote ion heating, and
      negative values denote ion cooling (electron heating).
    p_impurity_radiation_e: Impurity radiation power density
      [:math:`\mathrm{MW/m^3}`].
    Q_fusion: Fusion power gain (dimensionless).
    s_gas_puff: Gas puff particle source density [:math:`\mathrm{10^{20} m^{-3}
      s^{-1}}`].
    s_generic_particle: Generic particle source density [:math:`\mathrm{10^{20}
      m^{-3} s^{-1}}`].
    s_pellet: Pellet particle source density [:math:`\mathrm{10^{20} m^{-3}
      s^{-1}}`].
    Ip_profile: Total plasma current [:math:`\mathrm{MA}`].
    I_bootstrap: Total bootstrap current [:math:`\mathrm{MA}`].
    I_aux_generic: Total generic current source [:math:`\mathrm{MA}`].
    I_ecrh: Total electron cyclotron current [:math:`\mathrm{MA}`].
    P_auxiliary: Total auxiliary heating power [:math:`\mathrm{MW}`].
    P_ohmic_e: Total Ohmic heating power [:math:`\mathrm{MW}`].
    P_alpha_total: Total fusion alpha heating power [:math:`\mathrm{MW}`].
    P_sink: Total electron heating sink [:math:`\mathrm{MW}`].
    P_bremsstrahlung_e: Total bremsstrahlung radiation power loss
      [:math:`\mathrm{MW}`].
    P_cyclotron_e: Total cyclotron radiation power loss [:math:`\mathrm{MW}`].
    P_radiation_e: Total impurity radiation power loss [:math:`\mathrm{MW}`].
    t: Simulation time [:math:`\mathrm{s}`].
    rho_norm: Normalized toroidal flux coordinate on cell grid + boundaries.
    rho_cell_norm: Normalized toroidal flux coordinate on the cell grid.
    rho_face_norm: Normalized toroidal flux coordinate on the face grid.
    T_e_volume_avg: Volume-averaged electron temperature [:math:`\mathrm{keV}`].
    T_i_volume_avg: Volume-averaged ion temperature [:math:`\mathrm{keV}`].
    n_e_volume_avg: Volume-averaged electron density [:math:`\mathrm{10^{20}
      m^{-3}}`].
    n_i_volume_avg: Volume-averaged ion density [:math:`\mathrm{10^{20}
      m^{-3}}`].
    W_thermal_total: Total thermal stored energy [:math:`\mathrm{MJ}`].
    q95: Safety factor at 95% of the normalized poloidal flux.
    chi_total_i: Total ion heat conductivity (chi_turb_i + chi_neo_i)
      [:math:`\mathrm{m^2/s}`] on the face grid.
    chi_total_e: Total electron heat conductivity (chi_turb_e + chi_neo_e)
      [:math:`\mathrm{m^2/s}`] on the face grid.
    D_total_e: Total electron particle diffusivity (D_turb_e + D_neo_e)
      [:math:`\mathrm{m^2/s}`] on the face grid.
    V_total_e: Total electron particle convection (V_turb_e + V_neo_e +
      V_neo_ware_e) [:math:`\mathrm{m^2/s}`] on the face grid.
    V_neo_total_e: Neoclassical electron particle convection
      [:math:`\mathrm{m^2/s}`] on the face grid. Contains all components
      including the Ware pinch.
  """

  T_i: np.ndarray
  T_e: np.ndarray
  n_e: np.ndarray
  n_i: np.ndarray
  n_impurity: np.ndarray
  Z_impurity: np.ndarray
  psi: np.ndarray
  v_loop: np.ndarray
  j_total: np.ndarray
  j_ohmic: np.ndarray
  j_bootstrap: np.ndarray
  j_ecrh: np.ndarray
  j_generic_current: np.ndarray
  j_external: np.ndarray
  q: np.ndarray
  magnetic_shear: np.ndarray
  chi_turb_i: np.ndarray
  chi_neo_i: np.ndarray
  chi_turb_e: np.ndarray
  chi_neo_e: np.ndarray
  D_turb_e: np.ndarray
  D_neo_e: np.ndarray
  V_turb_e: np.ndarray
  V_neo_e: np.ndarray
  V_neo_ware_e: np.ndarray
  p_icrh_i: np.ndarray
  p_icrh_e: np.ndarray
  p_generic_heat_i: np.ndarray
  p_generic_heat_e: np.ndarray
  p_ecrh_e: np.ndarray
  p_alpha_i: np.ndarray
  p_alpha_e: np.ndarray
  p_ohmic_e: np.ndarray
  p_bremsstrahlung_e: np.ndarray
  p_cyclotron_radiation_e: np.ndarray
  ei_exchange: np.ndarray
  p_impurity_radiation_e: np.ndarray
  Q_fusion: np.ndarray  # pylint: disable=invalid-name
  s_gas_puff: np.ndarray
  s_generic_particle: np.ndarray
  s_pellet: np.ndarray
  Ip_profile: np.ndarray
  I_bootstrap: np.ndarray
  I_aux_generic: np.ndarray
  I_ecrh: np.ndarray
  P_auxiliary: np.ndarray
  P_ohmic_e: np.ndarray
  P_alpha_total: np.ndarray
  P_sink: np.ndarray
  P_bremsstrahlung_e: np.ndarray
  P_cyclotron_e: np.ndarray
  P_radiation_e: np.ndarray
  t: np.ndarray
  rho_norm: np.ndarray
  rho_cell_norm: np.ndarray
  rho_face_norm: np.ndarray
  T_e_volume_avg: np.ndarray
  T_i_volume_avg: np.ndarray
  n_e_volume_avg: np.ndarray
  n_i_volume_avg: np.ndarray
  W_thermal_total: np.ndarray  # pylint: disable=invalid-name
  q95: np.ndarray

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
        output.J_TOTAL: 1e6,  # A/m^2 to MA/m^2
        output.J_OHMIC: 1e6,  # A/m^2 to MA/m^2
        output.J_BOOTSTRAP: 1e6,  # A/m^2 to MA/m^2
        output.J_EXTERNAL: 1e6,  # A/m^2 to MA/m^2
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

  return PlotData(
      T_i=profiles_dataset[output.T_I].to_numpy(),
      T_e=profiles_dataset[output.T_E].to_numpy(),
      n_e=profiles_dataset[output.N_E].to_numpy(),
      n_i=profiles_dataset[output.N_I].to_numpy(),
      n_impurity=profiles_dataset[output.N_IMPURITY].to_numpy(),
      Z_impurity=profiles_dataset[output.Z_IMPURITY].to_numpy(),
      psi=profiles_dataset[output.PSI].to_numpy(),
      v_loop=profiles_dataset[output.V_LOOP].to_numpy(),
      j_total=profiles_dataset[output.J_TOTAL].to_numpy(),
      j_ohmic=profiles_dataset[output.J_OHMIC].to_numpy(),
      j_bootstrap=profiles_dataset[output.J_BOOTSTRAP].to_numpy(),
      j_external=profiles_dataset[output.J_EXTERNAL].to_numpy(),
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
  plotdata_fields = set(plotdata1.__dataclass_fields__)
  plotdata_properties = {
      name
      for name, _ in inspect.getmembers(
          type(plotdata1), lambda o: isinstance(o, property)
      )
  }
  plotdata_attrs = plotdata_fields.union(plotdata_properties)
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
