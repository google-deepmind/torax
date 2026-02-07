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

from torax._src.plotting import plotruns_lib

PLOT_CONFIG = plotruns_lib.FigureProperties(
    rows=4,
    cols=4,
    tick_size=8,
    subplot_title_size=12,
    height=None,
    font_family='Arial, sans-serif',
    title_size=16,
    axes=(
        # For chi, set histogram percentile for y-axis upper limit, due to
        # volatile nature of the data. Do not include first timepoint since
        # chi is defined as zero there and may unduly affect ylim.
        plotruns_lib.PlotProperties(
            attrs=('T_i', 'T_e'),
            labels=(r'$T_\mathrm{i}$', r'$T_\mathrm{e}$'),
            ylabel='Temperature [keV]',
        ),
        plotruns_lib.PlotProperties(
            attrs=('n_e', 'n_i'),
            labels=(r'$n_\mathrm{e}$', r'$n_\mathrm{i}$'),
            ylabel=r'Density $[10^{20}~m^{-3}]$',
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'chi_total_i',
                'chi_total_e',
            ),
            labels=(
                r'$\chi_\mathrm{i}$',
                r'$\chi_\mathrm{e}$',
            ),
            ylabel=r'Heat conductivity $[m^2/s]$',
            upper_percentile=98.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'D_total_e',
                'V_total_e',
            ),
            labels=(
                r'$D_\mathrm{e}$',
                r'$V_\mathrm{e}$',
            ),
            ylabel=r'Diff $[m^2/s]$ or Conv $[m/s]$',
            upper_percentile=98.0,
            lower_percentile=2.0,
            include_first_timepoint=False,
            ylim_min_zero=False,
        ),
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
            attrs=('psi',),
            labels=(r'$\psi$',),
            ylabel=r'Poloidal flux [Wb]',
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
                r'$j_\mathrm{tot}$',
                r'$j_\mathrm{ohm}$',
                r'$j_\mathrm{bs}$',
                r'$j_\mathrm{generic}$',
                r'$j_\mathrm{ecrh}$',
            ),
            ylabel=r'Toroidal current $[MA~m^{-2}]$',
            legend_fontsize=7,  # Smaller fontsize for this plot
            suppress_zero_values=True,  # Do not plot all-zero data
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
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=('Q_fusion',),
            labels=(r'$Q_\mathrm{fusion}$',),
            ylabel='Fusion gain',
        ),
        plotruns_lib.PlotProperties(
            attrs=('v_loop',),
            labels=(r'$\dot{\psi}$',),
            ylabel='Loop voltage',
            upper_percentile=98.0,
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'p_icrh_i',
                'p_icrh_e',
                'p_ecrh_e',
                'p_generic_heat_i',
                'p_generic_heat_e',
            ),
            labels=(
                r'$Q_\mathrm{ICRH,i}$',
                r'$Q_\mathrm{ICRH,e}$',
                r'$Q_\mathrm{ECRH,e}$',
                r'$Q_\mathrm{generic,i}$',
                r'$Q_\mathrm{generic,e}$',
            ),
            ylabel=r'External heat source density $[MW~m^{-3}]$',
            legend_fontsize=7,  # Smaller fontsize for this plot
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=('p_alpha_i', 'p_alpha_e', 'p_ohmic_e', 'ei_exchange'),
            labels=(
                r'$Q_\mathrm{alpha,i}$',
                r'$Q_\mathrm{alpha,e}$',
                r'$Q_\mathrm{ohmic}$',
                r'$Q_\mathrm{ei}$',
            ),
            ylabel=r'Internal heat source density $[MW~m^{-3}]$',
            legend_fontsize=6,  # Smaller fontsize for this plot
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=(
                'p_bremsstrahlung_e',
                'p_impurity_radiation_e',
                'p_cyclotron_radiation_e',
            ),
            labels=(
                r'$Q_\mathrm{brems}$',
                r'$Q_\mathrm{rad}$',
                r'$Q_\mathrm{cycl}$',
            ),
            ylabel=r'Heat sink density $[MW~m^{-3}]$',
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            plot_type=plotruns_lib.PlotType.TIME_SERIES,
            attrs=(
                'P_auxiliary',
                'P_ohmic_e',
                'P_alpha_total',
                'P_bremsstrahlung_e',
                'P_radiation_e',
                'P_cyclotron_e',
            ),
            labels=(
                r'$P_\mathrm{aux}$',
                r'$P_\mathrm{ohm}$',
                r'$P_\mathrm{\alpha}$',
                r'$P_\mathrm{brems}$',
                r'$P_\mathrm{rad}$',
                r'$P_\mathrm{cycl}$',
            ),
            ylabel=r'Total heating/sink powers $[MW]$',
            legend_fontsize=6,  # Smaller fontsize for this plot
            suppress_zero_values=True,  # Do not plot all-zero data
        ),
        plotruns_lib.PlotProperties(
            attrs=('Z_impurity',),
            labels=(r'$\langle Z_{impurity} \rangle$',),
            ylabel=r'Average impurity charge [amu]',
        ),
    ),
)
