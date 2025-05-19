.. _plotting:

Plotting simulations
####################

TORAX provides plotting functionalities to visualize simulation outputs and
compare different runs. The plots include a time slider for interactive
manipulation.

Using the plot_torax Script
============================

To visualize simulation results, use the ``plot_torax`` script.
This script offers flexibility in plotting single runs, comparing two runs, and
customizing plot configurations.

Plotting a Single Run
---------------------

To plot the output of a single simulation, run the following command from the
TORAX root directory:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output> \
   --plot_config <path_to_plot_config>

Replace ``<full_path_to_simulation_output>`` with the full path to your
simulation's output file. Optionally, specify a custom plot configuration module
using ``--plot_config``, with the path for the Python file containing the
``PLOT_CONFIG`` variable. See :ref:`customizable_plot_configs` for more
details.

Comparing Two Runs
------------------

To compare the output of two simulations, provide both output file paths:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output1> \
   <full_path_to_simulation_output2>

This overlays the plots of the two runs, allowing for direct comparison.

.. _customizable_plot_configs:

Customizable Plot Configurations
================================

The ``--plot_config`` flag allows you to define which quantities are plotted and
how they are arranged. This flag accepts the path to a Python file containing
a ``PLOT_CONFIG`` variable which is an instance of
``torax.plotting.plotruns_lib.FigureProperties``. This variable specifies the
layout and properties of the plot. If no ``--plot_config`` is specified, the
default configuration
`default_plot_config.py <https://github.com/google-deepmind/torax/tree/main/torax/plotting/configs/default_plot_config.py>`_.
is used.

Prepackaged Plot Configurations
-------------------------------

Several pre-packaged plot configurations are available in the
``plotting/configs`` directory. These include:

``default_plot_config.py`` - A default configuration with various profiles and
scalars including kinetic profiles, heating and current source profiles,
transport coefficients, and post-processed output quantities.

``simple_plot_config.py`` - A simplified configuration containing a subset of
the default configuration. Plots kinetic profiles, currents, transport
coefficients, safety factor and magnetic shear.

``sources_plot_config.py`` - A configuration focusing on profile and scalar
plots of heat, particle and current source profiles and their integrated
quantities.

``global_params_plot_config.py`` - A configuration focusing on scalar parameters
and their evolution over time, to provide a quick overview of the simulation.
Includes total currents, fusion gain, averaged kinetic profiles, q95.

Example usage:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output> \
   --plot_config plotting/configs/sources_plot_config.py

These plot configurations can also be passed as input into the ``run_torax``
script using the ``--plot_config`` flag. See :ref:`torax_flags`.

The ``PLOT_CONFIG`` from ``plotting/configs/simple_plot_config.py`` is shown
below as an example.

.. code-block:: python

  from torax.plotting import plotruns_lib

  PLOT_CONFIG = plotruns_lib.FigureProperties(
      rows=2,
      cols=3,
      axes=(
          plotruns_lib.PlotProperties(
              attrs=('T_i', 'T_e'),
              labels=(r'$T_i$', r'$T_e$'),
              ylabel='Temperature [keV]',
          ),
          plotruns_lib.PlotProperties(
              attrs=('n_e',),
              labels=(r'$n_e$',),
              ylabel=r'Electron density $[10^{20}~m^{-3}]$',
          ),
          plotruns_lib.PlotProperties(
              attrs=('chi_turb_i', 'chi_turb_e'),
              labels=(r'$\chi_i$', r'$\chi_e$'),
              ylabel=r'Heat conductivity $[m^2/s]$',
              upper_percentile=98.0,
              include_first_timepoint=False,
              ylim_min_zero=False,
          ),
          plotruns_lib.PlotProperties(
              attrs=(
                  'j_total',
                  'j_ohmic',
                  'j_bootstrap',
                  'j_generic_current',
                  'j_ecrh',
              ),
              labels=(
                  r'$j_{tot}$',
                  r'$j_{ohm}$',
                  r'$j_{bs}$',
                  r'$j_{generic}$',
                  r'$j_{ecrh}$',
              ),
              ylabel=r'Toroidal current $[MA~m^{-2}]$',
              legend_fontsize=8,
          ),
          plotruns_lib.PlotProperties(
              attrs=('q',),
              labels=(r'$q$',),
              ylabel='Safety factor',
          ),
          plotruns_lib.PlotProperties(
              attrs=('magnetic_shear',),
              labels=(r'$\hat{s}$',),
              ylabel='Magnetic shear',
          ),
      ),
  )


Customizing Plots
-----------------

The ``FigureProperties`` dataclass offers several options for customizing the
plot layout and content. Dataclass fields and defaults are as follows:

- ``rows`` (int): Number of rows in the figure.
- ``cols`` (int): Number of columns in the figure.
- ``axes`` (tuple of ``PlotProperties``):  Configuration for each subplot.
  See below.
- ``figure_size_factor`` (float=5.0): Scaling factor for the figure size.
- ``tick_fontsize`` (int=10): Font size for axis ticks.
- ``axes_fontsize`` (int=10): Font size for axis labels.
- ``title_fontsize`` (int=16): Font size for the figure title.
- ``default_legend_fontsize`` (int=10): Default font size for legends.
- ``colors`` (tuple[str, ...] = ('r', 'b', 'g', 'm', 'y', 'c')): Colors to use
  for plot lines. Cycles through the tuple for multiple lines.

The ``PlotProperties`` dataclass configures individual subplots. For example,
the ``PlotProperties`` object for plotting ion and electron temperatures looks
like this:

.. code-block:: python

  plotruns_lib.PlotProperties(
      attrs=('T_i', 'T_e'),
      labels=(r'$T_i$', r'$T_e$'),
      ylabel='Temperature [keV]',
  ),


The fields in ``PlotProperties`` are as follows:

- ``attrs``: Tuple of attribute names from the ``PlotData`` dataclass used to
  retrieve the data for plotting.
- ``labels``: Tuple of labels for the plotted lines, one label per attribute in
  ``attrs``.
- ``ylabel``: Label for the y-axis.
- ``legend_fontsize`` (int | None): Legend font size. If None, defaults to
  ``default_legend_fontsize`` in ``FigureProperties``.
- ``upper_percentile`` (float=100.0): Filters out outlier data above a given
  percentile for plotting purposes.
- ``lower_percentile`` (float=0.0): Filters out outlier data below a given
  percentile for plotting purposes.
- ``include_first_timepoint`` (bool=True): Whether to include the first time
  point in calculating plot range.
- ``ylim_min_zero`` (bool=True): Whether the plot limits should start from zero.
- ``plot_type`` (PlotType=PlotType.SPATIAL): Defines whether the plot is a
  spatial profile, or time series plot.
- ``suppress_zero_values`` (bool=False): If True, all-zero-data is not plotted.

``suppress_zero_values`` is useful when defining plots where not all the
``attrs`` may be relevant for all runs. For example, if a run does not include a
bootstrap current, the ``j_bootstrap`` attribute will be all zero. Setting
``suppress_zero_values=True`` will automatically exclude this line from the
plot.

``upper_percentile`` and ``lower_percentile`` are useful for excluding outliers
from the plot range calculation, for example transient spikes in the data.

``plot_type`` can be set to either ``PlotType.SPATIAL`` (default) or
``PlotType.TIME_SERIES``. Spatial plots are 1D profiles which are updated at
each time slice, following time slider manipulation. Time series plots are 0D
quantities plotted against the full simulation time, and are not affected by the
time slider.

By creating a custom Python module with a ``PLOT_CONFIG`` variable set to a
FigureProperties instance you can thus completely customize which variables are
plotted by Torax by defining a new FigureProperties instance. For examples, see
torax/plotting/configs/\*.py.

Interactive Time Slider
=======================

When plotting with ``plot_torax``, an interactive time slider appears below the
plots. This slider allows you to scroll through the simulation output at
different timesteps. The spatial profile plots defined above are dynamically
updated when the slider is manipulated. The time series plots, defined with
``plot_type=PlotType.TIME_SERIES`` stay constant, plotting variables against the
full Torax simulation timescale.
