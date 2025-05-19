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

"""Simple plotting configuration for Torax runs."""

from torax._src.plotting import plotruns_lib

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
            upper_percentile=98.0,  # Exclude outliers
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
            legend_fontsize=8,  # Smaller fontsize for this plot
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
