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

"""Plotting configuration for Torax runs focusing on transport coefficients."""

from torax._src.plotting import plotruns_lib

PLOT_CONFIG = plotruns_lib.FigureProperties(
    rows=2,
    cols=3,
    tick_fontsize=10,
    axes_fontsize=10,
    default_legend_fontsize=9,
    figure_size_factor=5,
    title_fontsize=12,
    axes=(
        plotruns_lib.PlotProperties(
            attrs=(
                'chi_turb_i',
                'chi_turb_e',
            ),
            labels=(
                r'$\chi_\mathrm{turb,i}$',
                r'$\chi_\mathrm{turb,e}$',
            ),
            ylabel=r'Turbulent heat conductivity $[m^2/s]$',
            upper_percentile=98.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
        plotruns_lib.PlotProperties(
            attrs=('D_turb_e',),
            labels=(r'$D_\mathrm{turb,e}$',),
            ylabel=r'Turbulent particle diffusivity $[m^2/s]$$',
            upper_percentile=98.0,
            lower_percentile=2.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
        plotruns_lib.PlotProperties(
            attrs=('V_turb_e',),
            labels=(r'$V_\mathrm{turb,e}$',),
            ylabel=r'Turbulent particle convectivity $[m^2/s]$$',
            upper_percentile=98.0,
            lower_percentile=2.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'chi_neo_i',
                'chi_neo_e',
            ),
            labels=(
                r'$\chi_\mathrm{neo,i}$',
                r'$\chi_\mathrm{neo,e}$',
            ),
            ylabel=r'Neoclassical heat conductivity $[m^2/s]$',
            upper_percentile=98.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
            suppress_zero_values=True,
        ),
        plotruns_lib.PlotProperties(
            attrs=('D_neo_e',),
            labels=(r'$D_\mathrm{neo,e}$',),
            ylabel=r'Neoclassical particle diffusivity $[m^2/s]$$',
            upper_percentile=98.0,
            lower_percentile=2.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
            suppress_zero_values=True,
        ),
        plotruns_lib.PlotProperties(
            attrs=('V_neo_tot_e', 'V_neo_e', 'V_neo_ware_e'),
            labels=(
                r'$V_\mathrm{neo,e}+V_\mathrm{ware,e}$',
                r'$V_\mathrm{neo,e}$',
                r'$V_\mathrm{ware,e}$',
            ),
            ylabel=r'Neoclassical particle convectivity $[m^2/s]$$',
            upper_percentile=98.0,
            lower_percentile=2.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
            suppress_zero_values=True,
        ),
    ),
)
