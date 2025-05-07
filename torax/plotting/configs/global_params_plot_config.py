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

"""Plotting configuration for global plasma parameters vs time."""

from torax.plotting import plotruns_lib

PLOT_CONFIG = plotruns_lib.FigureProperties(
    rows=2,
    cols=3,
    tick_fontsize=10,
    axes_fontsize=10,
    default_legend_fontsize=8,
    figure_size_factor=3,
    title_fontsize=12,
    axes=(
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('Ip_profile', 'I_bootstrap', 'I_aux_generic', 'I_ecrh'),
            labels=(
                r'$I_\mathrm{p}$',
                r'$I_\mathrm{bs}$',
                r'$I_\mathrm{generic}$',
                r'$I_\mathrm{ecrh}$',
            ),
            ylabel=r'Current [MA]',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('Q_fusion',),
            labels=(r'$Q_\mathrm{fusion}$',),
            ylabel='Fusion gain',
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('W_thermal_total',),
            labels=(r'$W_\mathrm{therm\_tot}$',),
            ylabel='Total thermal stored energy [MJ]',
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('T_e_volume_avg', 'T_i_volume_avg'),
            labels=(
                r'$\mathrm{<T_e>_V}$',
                r'$\mathrm{<T_i>_V}$',
            ),
            ylabel='Volume average T_e and T_i [keV]',
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('n_e_volume_avg', 'n_i_volume_avg'),
            labels=(
                r'$\mathrm{<n_e>_V}$',
                r'$\mathrm{<n_i>_V}$',
            ),
            ylabel='Volume average ne and ni $[10^{20}~m^{-3}]$',
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('q95',),
            labels=(r'$q_\mathrm{95}$',),
            ylabel='q at 95% of the normalised psi',
        ),
    ),
)
