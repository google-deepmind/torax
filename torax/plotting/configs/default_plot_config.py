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

"""Default plotting configuration for Torax runs."""

from torax.plotting import plotruns_lib

PLOT_CONFIG = plotruns_lib.FigureProperties(
    rows=2,
    cols=3,
    axes=(
        # For chi, set histogram percentile for y-axis upper limit, due to
        # volatile nature of the data. Do not include first timepoint since
        # chi is defined as zero there and may unduly affect ylim.
        plotruns_lib.PlotProperties(
            attrs=('chi_i', 'chi_e'),
            labels=(r'$\chi_i$', r'$\chi_e$'),
            ylabel=r'Heat conductivity $[m^2/s]$',
            upper_percentile=98.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
        plotruns_lib.PlotProperties(
            attrs=('ti', 'te'),
            labels=(r'$T_i$', r'$T_e$'),
            ylabel='Temperature [keV]',
        ),
        plotruns_lib.PlotProperties(
            attrs=('ne',),
            labels=(r'$n_e$',),
            ylabel=r'Electron density $[10^{20}~m^{-3}]$',
        ),
        plotruns_lib.PlotProperties(
            attrs=('j', 'johm', 'j_bootstrap', 'jext'),
            labels=(r'$j_{tot}$', r'$j_{ohm}$', r'$j_{bs}$', r'$j_{ext}$'),
            ylabel=r'Toroidal current $[A~m^{-2}]$',
            legend_fontsize=8,  # Smaller fontsize for this plot
        ),
        plotruns_lib.PlotProperties(
            attrs=('q',),
            labels=(r'$q$',),
            ylabel='Safety factor',
        ),
        plotruns_lib.PlotProperties(
            attrs=('s',),
            labels=(r'$\hat{s}$',),
            ylabel='Magnetic shear',
        ),
    ),
)
