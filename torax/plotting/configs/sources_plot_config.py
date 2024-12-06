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

"""Plotting configuration for Torax runs focusing on source profiles."""

from torax.plotting import plotruns_lib

PLOT_CONFIG = plotruns_lib.FigureProperties(
    rows=2,
    cols=4,
    tick_fontsize=10,
    axes_fontsize=10,
    default_legend_fontsize=9,
    figure_size_factor=5,
    title_fontsize=12,
    axes=(
        # For chi, set histogram percentile for y-axis upper limit, due to
        # volatile nature of the data. Do not include first timepoint since
        # chi is defined as zero there and may unduly affect ylim.
        plotruns_lib.PlotProperties(
            attrs=('ti', 'te'),
            labels=(r'$T_\mathrm{i}$', r'$T_\mathrm{e}$'),
            ylabel='Temperature [keV]',
        ),
        plotruns_lib.PlotProperties(
            attrs=('ne',),
            labels=(r'$n_\mathrm{e}$',),
            ylabel=r'Electron density $[10^{20}~m^{-3}]$',
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'j',
                'johm',
                'j_bootstrap',
                'generic_current_source',
                'j_ecrh',
            ),
            labels=(
                r'$j_\mathrm{tot}$',
                r'$j_\mathrm{ohm}$',
                r'$j_\mathrm{bs}$',
                r'$j_\mathrm{generic}$',
                r'$j_\mathrm{ecrh}$',
            ),
            ylabel=r'Toroidal current $[MA~m^{-2}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('p_auxiliary', 'p_ohmic', 'p_alpha', 'p_sink'),
            labels=(
                r'$P_\mathrm{aux}$',
                r'$P_\mathrm{ohm}$',
                r'$P_\mathrm{\alpha}$',
                r'$P_\mathrm{sink}$',
            ),
            ylabel=r'Total heating/sink powers $[MW]$',
            legend_fontsize=8,  # Smaller fontsize for this plot
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'q_icrh_i',
                'q_icrh_e',
                'q_ecrh',
                'q_gen_i',
                'q_gen_e',
            ),
            labels=(
                r'$Q_\mathrm{ICRH,i}$',
                r'$Q_\mathrm{ICRH,e}$',
                r'$Q_\mathrm{ECRH,e}$',
                r'$Q_\mathrm{generic,i}$',
                r'$Q_\mathrm{generic,e}$',
            ),
            ylabel=r'External heat source density $[MW~m^{-3}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=('q_alpha_i', 'q_alpha_e', 'q_ohmic', 'q_ei'),
            labels=(
                r'$Q_\mathrm{alpha,i}$',
                r'$Q_\mathrm{alpha,e}$',
                r'$Q_\mathrm{ohmic}$',
                r'$Q_\mathrm{ei}$',
            ),
            ylabel=r'Internal heat source density $[MW~m^{-3}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=('q_brems', 'q_rad'),
            labels=(r'$Q_\mathrm{brems}$', r'$Q_\mathrm{rad}$'),
            ylabel=r'Heat sink density $[MW~m^{-3}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=('s_puff', 's_generic', 's_pellet'),
            labels=(
                r'$S_\mathrm{puff}$',
                r'$S_\mathrm{generic}$',
                r'$S_\mathrm{pellet}$',
            ),
            ylabel=r'Particle sources $[10^{20}~m^{-3}~s^{-1}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
    ),
)
