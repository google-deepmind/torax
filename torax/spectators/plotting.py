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

"""Module for plotting tools."""

import dataclasses
import math
from typing import Any, Callable, Sequence

from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from torax import geometry
from torax.spectators import spectator


@dataclasses.dataclass
class PlotOptions:
  # Axis labels.
  x_label: str | None = None
  y_label: str | None = None


@dataclasses.dataclass
class PlotKey:
  """Defines a line to plot in one subplot."""

  # Key to look up in the observed set of JAX arrays. See
  # InMemoryJaxArraySpectator for an example of what keys this can be used with.
  key: str

  # X-axis data.
  # While plotting, the X-axis will not change but rather remain fixed with
  # these values throughout the duration of the sim and therefore the duration
  # of the plots.
  x_axis: jnp.ndarray | np.ndarray

  # Label to apply to the data.
  label: str | None = None

  # Data transform to apply to all data plotted on the y-axis.
  y_data_transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None


@dataclasses.dataclass
class Plot:
  """Configuration for a single plot."""

  # Defines the values that should be plotted.
  keys: Sequence[PlotKey]

  # Which colors each key should be plotted with.
  colors: Sequence[str]

  # Extra options for the plot.
  plot_options: PlotOptions

  def __post_init__(self):
    if len(self.keys) != len(self.colors):
      raise ValueError(
          'Must provide the same number of keys as colors. '
          f'Given keys: {self.keys}. Given colors: {self.colors}.'
      )


class Plotter:
  """Manages plotting to a matplotlib figure."""

  def __init__(
      self,
      plots: Sequence[Plot],
      pyplot_figure_kwargs: dict[str, Any] | None = None,
      max_plots_in_row: int = 2,
  ):
    """Initializes the Plotter.

    The Plotter will create a matplotlib figure on initialization (and whenever
    reset() is called). The arguments to the __init__ define what is shown in
    that newly created figure.

    Args:
      plots: Sequence of plot configurations which each define what will be
        plotted. Each of these plots will be a subplot in the matplotlib Figure
        this Plotter creates.
      pyplot_figure_kwargs: Extra kwargs to pass into the matplotlib Figure
        constructor.
      max_plots_in_row: Number of subplots to have in each row in the figure.
    """
    self._plots = plots
    self._pylot_figure_kwargs = pyplot_figure_kwargs or {}
    self._max_plots_in_row = max_plots_in_row
    self.reset()

  @property
  def figure(self) -> plt.Figure:
    """Returns the matplotlib figure currently associated with this Plotter."""
    return self._figure

  def close(self) -> None:
    if hasattr(self, '_figure'):
      plt.close(self._figure)

  def reset(self):
    """Resets the plotter and creates a new figure to plot on."""
    self.close()
    fig = plt.figure(**self._pylot_figure_kwargs)
    num_cols = self._max_plots_in_row
    num_rows = int(math.ceil(len(self._plots) / num_cols))
    lines = []
    for idx, plot in enumerate(self._plots):
      axes = fig.add_subplot(num_rows, num_cols, idx + 1)
      axes.set_xlabel(plot.plot_options.x_label)
      axes.set_ylabel(plot.plot_options.y_label)
      axes_lines = []
      for key, color in zip(plot.keys, plot.colors):
        line = axes.plot(
            key.x_axis,
            jnp.zeros_like(key.x_axis),
            color=color,
            label=key.label,
        )
        axes_lines.append(line[0])
      lines.append(tuple(axes_lines))
      if len(plot.keys) > 1:
        axes.legend(fontsize=8)
    fig.tight_layout()
    self._figure = fig
    self._lines = tuple(lines)

  def update_data(
      self,
      data: dict[str, jnp.ndarray],
  ) -> None:
    """Updates the data drawn in the plots."""
    for plot, axes_lines in zip(self._plots, self._lines):
      for subplot, line in zip(plot.keys, axes_lines):
        if subplot.key not in data:
          continue
        line_data = data[subplot.key]
        if subplot.y_data_transform is not None:
          line_data = subplot.y_data_transform(line_data)
        line.set_ydata(line_data)
    # Rescale all the Y axes.
    for axes in self._figure.get_axes():
      axes_data = []
      for line in axes.get_lines():
        axes_data.append(line.get_ydata())
      axes.set_ylim(_get_y_min(axes_data), _get_y_max(axes_data))
    self._figure.canvas.draw()


class PlotSpectator(spectator.Spectator):
  """Spectator that runs with the simulator and plots state as it steps."""

  def __init__(
      self,
      plots: Sequence[Plot],
      pyplot_figure_kwargs: dict[str, Any] | None = None,
      max_plots_in_row: int = 2,
  ):
    """Initializes the PlotSpectator.

    The PlotSpectator will create a matplotlib Figure on initialization (and
    whenever reset() is called).

    Args:
      plots: Sequence of plot configurations which each define what will be
        plotted. Each of these plots will be a subplot in the matplotlib Figure.
      pyplot_figure_kwargs: Extra kwargs to pass into the matplotlib Figure
        constructor.
      max_plots_in_row: Number of subplots to have in each row in the figure.
    """
    self._spectator = spectator.InMemoryJaxArraySpectator()
    self._plotter = Plotter(
        plots=plots,
        pyplot_figure_kwargs=pyplot_figure_kwargs,
        max_plots_in_row=max_plots_in_row,
    )

  def reset(self) -> None:
    """Resets the observed history and creates a new plot to plot on."""
    self._plotter.reset()
    self._spectator.reset()

  def after_step(self):
    """Updates the figure shown and pauses for 10 ms."""
    self.update_plots()
    plt.pause(0.01)

  def observe(self, key: str, data: jnp.ndarray) -> None:
    self._spectator.observe(key, data)

  def update_plots(self):
    self.refresh_plots_at_new_timestep(-1)

  def refresh_plots_at_new_timestep(
      self,
      timestep_idx: int,
  ):
    data_at_timestep = spectator.get_data_at_index(
        self._spectator, timestep_idx
    )
    self._plotter.update_data(data_at_timestep)

  @property
  def figure(self) -> plt.Figure:
    return self._plotter.figure


def _get_y_min(
    data: Sequence[jnp.ndarray | np.ndarray],
) -> float:
  min_val = np.min(data)
  if min_val < 0:
    min_val = min_val * 1.1
  else:
    min_val = min_val / 1.1
  return min_val


def _get_y_max(
    data: Sequence[jnp.ndarray | np.ndarray],
) -> float:
  max_val = np.max(data)
  if max_val < 0:
    max_val = max_val / 1.1
  else:
    max_val = max_val * 1.1
  return max_val


def get_default_plot_config(
    geo: geometry.Geometry,
) -> Sequence[Plot]:
  """Returns the default plot configuration to run with the simulator.

  This defines which values will be plotted and how different lines are grouped
  together in subplots.

  NOTE: This assumes the simulation is running with implicit updates (otherwise
  the plots will likely miss most information to plot and won't be very
  interesting to look at).

  Args:
    geo: The geometry fed into the simulation run. Used for defining the X-axis
      for many of the subplots.

  Returns:
    Tuple of Plot objects which can be fed into a Plotter instance.
  """
  # Keep track of the transforms we want to apply to the data before plotting.
  data_transforms = {
      'source_ion': lambda arr: arr / 1e3,
      'Pfus_i': lambda arr: arr / 1e3,
      'Qei': lambda arr: arr / 1e3,
      'source_el': lambda arr: arr / 1e3,
      'Pfus_e': lambda arr: arr / 1e3,
      'jtot_face': lambda arr: arr / 1e6,
      'jext_face': lambda arr: arr / 1e6,
      'j_bootstrap_face': lambda arr: arr / 1e6,
      'johm_face': lambda arr: arr / 1e6,
  }

  def get_plot(
      keys: Sequence[str],
      x_axis: jnp.ndarray | np.ndarray,
      x_label: str,
      y_label: str,
      labels: Sequence[str] | None = None,
      custom_data_transforms: (
          dict[str, Callable[[jnp.ndarray], jnp.ndarray]] | None
      ) = None,
  ) -> Plot:
    plot_keys = []
    if labels is None:
      labels = keys
    for key, label in zip(keys, labels):
      if custom_data_transforms and key in custom_data_transforms:
        transform = custom_data_transforms[key]
      elif key in data_transforms:
        transform = data_transforms[key]
      else:
        transform = None
      plot_keys.append(
          PlotKey(
              key=key,
              x_axis=x_axis,
              label=label,
              y_data_transform=transform,
          )
      )
    colors = ['r', 'b', 'g', 'm', 'y', 'c'][: len(keys)]
    return Plot(
        keys=tuple(plot_keys),
        colors=tuple(colors),
        plot_options=PlotOptions(x_label=x_label, y_label=y_label),
    )

  return (
      get_plot(
          keys=('chi_face_ion', 'chi_face_el'),
          x_axis=geo.r_face_norm,
          x_label='Normalized radius',
          y_label=r'Heat conductivity $[m^2/s]$',
      ),
      get_plot(
          keys=('ne',),
          x_axis=geo.r_norm,
          x_label='Normalized radius',
          y_label=r'Density $[10^{20}~m^{-3}]$',
      ),
      get_plot(
          keys=('source_ion', 'Pfus_i', 'Qei'),
          x_axis=geo.r_norm,
          x_label='Normalized radius',
          y_label=r'Ion heat source $[kW~m^{-3}]$',
          labels=('External term', 'Fusion term', 'Ion-electron heat exchange'),
      ),
      get_plot(
          keys=('source_el', 'Pfus_e', 'Qei'),
          x_axis=geo.r_norm,
          x_label='Normalized radius',
          y_label=r'Electron heat source $[kW~m^{-3}]$',
          labels=('External term', 'Fusion term', 'Ion-electron heat exchange'),
          custom_data_transforms={
              'Qei': lambda arr: -arr / 1e3,  # Negate it.
          },
      ),
      get_plot(
          keys=('q_face',),
          x_axis=geo.r_face_norm,
          x_label='Normalized radius',
          y_label='q-profile',
      ),
      get_plot(
          keys=('s_face',),
          x_axis=geo.r_face_norm,
          x_label='Normalized radius',
          y_label='Magnetic shear',
      ),
      get_plot(
          keys=('jtot_face', 'jext_face', 'j_bootstrap_face', 'johm_face'),
          x_axis=geo.r_face_norm,
          x_label='Normalized radius',
          y_label=r'Current $[MA~m^{-2}]$',
          labels=(
              'Total current',
              'External current',
              'Bootstrap current',
              'Ohmic current',
          ),
      ),
      get_plot(
          keys=('temp_ion', 'temp_el'),
          x_axis=geo.r_norm,
          x_label='Normalized radius',
          y_label='Temperature [keV]',
          labels=(r'$T_i$', r'$T_e$'),
      ),
  )
