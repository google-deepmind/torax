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

import dataclasses
import enum
import inspect
import itertools
from os import path
import re
from typing import Final, Mapping, Optional, Set

import immutabledict
import numpy as np
from plotly import subplots
import plotly.colors as pcolors
import plotly.graph_objects as go
from torax._src.output_tools import output
import xarray as xr

# Internal import.
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
  font_family: str
  title_size: int
  subplot_title_size: int
  tick_size: int
  height: int | None

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


def _get_file_path(outfile: str) -> str:
  """Gets the absolute path to the file."""
  possible_paths = [outfile]
  path_check_fns = [path.exists]

  for path_check_fn, possible_path in zip(
      path_check_fns, possible_paths, strict=True
  ):
    if path_check_fn(possible_path):
      return possible_path

  raise ValueError(f'Could not find {outfile}. Tried {possible_paths}.')


def _get_title(path1, path2):
  names = [f'(1) {path.basename(path1)}']
  if path2:
    names.append(f'(2) {path.basename(path2)}')
  return '  &  '.join(names)


def plot_run(
    plot_config: FigureProperties,
    outfile: str,
    outfile2: str | None = None,
    interactive: bool = True,
) -> go.Figure:
  """Plots a single run or comparison of two runs."""
  outfile = _get_file_path(outfile)
  outfile2 = _get_file_path(outfile2) if outfile2 else None

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

  fig_title = _get_title(outfile, outfile2)
  fig = create_plotly_figure(plot_config, plotdata1, plotdata2, fig_title)
  if interactive:
    fig.show()

  return fig


def _latex_to_html(latex_str: str) -> str:
  """Convert 'text $math$' -> 'html_str'."""

  # 1. Strip math delimiters and tildes.
  html = latex_str.replace('$', '').replace('~', ' ')
  html = re.sub(r'\\mathrm', '', html)

  # 2. Handle Braced Subscripts/Superscripts.
  # Matches _{text} or ^{text}
  html = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', html)
  html = re.sub(r'\^\{([^}]*)\}', r'<sup>\1</sup>', html)

  # 3. Handle Single Character Subscripts/Superscripts.
  # Matches _x or ^x (where x is any alphanumeric)
  html = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', html)
  html = re.sub(r'\^([a-zA-Z0-9])', r'<sup>\1</sup>', html)

  # 4. Greek Letter Map.
  greek_map = {r'\psi': 'ψ', r'\chi': 'χ'}

  for latex_char, html_entity in greek_map.items():
    html = html.replace(latex_char, html_entity)

  html = re.sub(r'\\dot\{([^}]*)\}', r'\1&#775;', html)
  html = re.sub(r'\\hat\{([^}]*)\}', r'\1&#770;', html)
  html = re.sub(r'\\langle', '<', html)
  html = re.sub(r'\\rangle', '>', html)
  return html


def _transform_string(input_str: str) -> str:
  """Convert 'text $math$' -> 'html_str'."""

  match_grp = re.match(
      r'^(?!(?:\s*)\$[^$]+\$(?:\s*)$)(?=.*[a-zA-Z0-9]).*\$.+\$.*$', input_str
  )
  if match_grp:
    return _latex_to_html(input_str)
  else:
    return input_str


def _subplot_prefix(idx: int):
  return f'({chr(97 + idx)})'


def _setup_subplots(
    plot_config: FigureProperties,
    axes: list[PlotProperties],
) -> go.Figure:
  """Setup subplots for the figure."""

  subplot_titles = []
  rows, cols = plot_config.rows, plot_config.cols
  for i in range(rows * cols):
    if i < len(axes):
      is_spatial = axes[i].plot_type == PlotType.SPATIAL
      x_label = 'Radius (rho)' if is_spatial else 'Time [s]'

      # Using ylabel as title and converting LaTeX to HTML
      prefix = _subplot_prefix(i)
      subplot_title = (
          f'{prefix} {_transform_string(axes[i].ylabel)} vs {x_label}'
      )
      subplot_titles.append(subplot_title)
    else:
      subplot_titles.append('')

  fig = subplots.make_subplots(
      rows=rows,
      cols=cols,
      subplot_titles=subplot_titles,
  )
  fig.update_annotations(
      font=dict(
          family=plot_config.font_family, size=plot_config.subplot_title_size
      )
  )
  return fig


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


def get_y_limits(data1, data2, cfg):
  """Gets the y limits for a set of attributes."""
  attrs, lower_percentile, upper_percentile, include_first_timepoint = (
      cfg.attrs,
      cfg.lower_percentile,
      cfg.upper_percentile,
      cfg.include_first_timepoint,
  )
  ymin = get_limit(data1, attrs, lower_percentile, include_first_timepoint)
  ymax = get_limit(data1, attrs, upper_percentile, include_first_timepoint)

  if data2:
    ymin = min(
        ymin, get_limit(data2, attrs, lower_percentile, include_first_timepoint)
    )
    ymax = max(
        ymax, get_limit(data2, attrs, upper_percentile, include_first_timepoint)
    )

  lower_bound = ymin / 1.05 if ymin > 0 else ymin * 1.05
  if cfg.ylim_min_zero:
    lower_bound = min(lower_bound, 0)
  if lower_bound == 0:
    lower_bound = -0.1
  upper_bound = ymax * 1.05

  return lower_bound, upper_bound


def create_plotly_figure(
    plot_config: FigureProperties,
    data1: PlotData,
    data2: Optional[PlotData] = None,
    title: str = 'Torax Simulation Results',
) -> go.Figure:
  """Create a plotly figure."""

  axes = sorted(
      plot_config.axes, key=lambda ax: ax.plot_type == PlotType.SPATIAL
  )

  fig = _setup_subplots(plot_config, axes)
  spatial_traces_info = []
  trace_count = 0
  colors = itertools.cycle(pcolors.qualitative.Plotly)
  datasets = [d for d in [data1, data2] if d is not None]

  # Add Traces
  for i, axis_config in enumerate(axes):
    row, col = (i // plot_config.cols) + 1, (i % plot_config.cols) + 1
    is_spatial = axis_config.plot_type == PlotType.SPATIAL
    prefix = _subplot_prefix(i)

    for attr, label in zip(axis_config.attrs, axis_config.labels):
      color = next(colors)

      for idx, dataset in enumerate(datasets):
        if axis_config.suppress_zero_values and np.all(
            getattr(dataset, attr) == 0
        ):
          continue

        if is_spatial:
          # Initial plot uses the first time step (index 0)
          x = get_rho(dataset, attr)
          y = getattr(dataset, attr)[0, :]

          # Record this trace so the slider can find it later
          spatial_traces_info.append(
              {'trace_idx': trace_count, 'attr': attr, 'dataset': dataset}
          )
        else:
          # Time series plots (static)
          x = dataset.t
          y = getattr(dataset, attr)

        label_html = f"{prefix} {_transform_string(f'{label} (Data {idx+1})')}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=label_html,
                showlegend=True,
                line=dict(color=color, dash='dash' if idx > 0 else 'solid'),
            ),
            row=row,
            col=col,
        )
        trace_count += 1

    # Update Axes Labels
    fig.update_xaxes(
        tickfont=dict(
            family=plot_config.font_family, size=plot_config.tick_size
        ),
        row=row,
        col=col,
    )
    ylow, yhigh = get_y_limits(data1, data2, axis_config)
    fig.update_yaxes(
        tickfont=dict(
            family=plot_config.font_family, size=plot_config.tick_size
        ),
        row=row,
        col=col,
        range=(ylow, yhigh),
    )

  # Build the Slider (Only if there are spatial plots)
  if spatial_traces_info:
    steps = []
    # Use data1 time as the master clock
    for t_idx, t_val in enumerate(data1.t):
      y_updates = []
      trace_indices = []

      for info in spatial_traces_info:
        # Check if this dataset actually has data for this time index
        if t_idx < len(info['dataset'].t):
          val_array = getattr(info['dataset'], info['attr'])
          y_updates.append(val_array[t_idx, :])
          trace_indices.append(info['trace_idx'])

      step = {
          'method': 'restyle',
          'label': f'{t_val:.3f}s',
          'args': [{'y': y_updates}, trace_indices],
      }
      steps.append(step)

    fig.update_layout(
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
            'pad': {'t': 50},
            'steps': steps,
        }]
    )

  # 5. Global Layout Update
  layout_args = {
      'title': {
          'text': title,
          'x': 0.5,
          'font': {'size': plot_config.title_size},
      },
      'margin': dict(l=40, r=40, t=80, b=40),
      'showlegend': True,
      'font': dict(family=plot_config.font_family),
  }

  if plot_config.height:
    layout_args['height'] = plot_config.height
  else:
    layout_args['autosize'] = True

  fig.update_layout(**layout_args)
  fig.update_annotations(
      font=dict(
          family=plot_config.font_family, size=plot_config.subplot_title_size
      )
  )

  return fig


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
