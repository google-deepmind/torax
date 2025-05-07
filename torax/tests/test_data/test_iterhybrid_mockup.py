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

"""ITER hybrid scenario approximately based on van Mulders NF 2021."""


CONFIG = {
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},  # (bundled isotope average)
        'impurity': 'Ne',
        'Zeff': 1.6,  # sets impurity density
    },
    'profile_conditions': {
        'Ip_tot': 10.5,  # total plasma current in MA
        # boundary + initial conditions for T and n
        # initial condition ion temperature for r=0 and r=a_minor
        'T_i': {0.0: {0.0: 15.0, 1.0: 0.2}},
        'T_i_right_bc': 0.2,  # boundary condition ion temperature for r=a_minor
        # initial condition electron temperature for r=0 and r=a_minor
        'T_e': {0.0: {0.0: 15.0, 1.0: 0.2}},
        'T_e_right_bc': 0.2,  # boundary condition electron temp for r=a_minor
        'n_e_right_bc': 0.25,  # boundary condition density for r=a_minor
        # set initial condition density according to Greenwald fraction.
        'n_e_nbar_is_fGW': True,
        'nbar': 0.8,
        'n_e': {0: {0.0: 1.5, 1.0: 1.0}},  # Initial electron density profile
    },
    'numerics': {
        # simulation control
        't_final': 5,  # length of simulation time in seconds
        # 1/multiplication factor for sigma (conductivity) to reduce current
        # diffusion timescale to be closer to heat diffusion timescale.
        'resistivity_multiplier': 200,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        # multiplier in front of the base timestep dt=dx^2/(2*chi). Can
        # likely be increased further beyond this default.
        'chi_timestep_prefactor': 50,
        'dt_reduction_factor': 3,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'Rmaj': 6.2,  # major radius (R) in meters
        'Rmin': 2.0,  # minor radius (a) in meters
        'B0': 5.3,  # Toroidal magnetic field on axis [T]
    },
    'sources': {
        # Current sources (for psi equation)
        'j_bootstrap': {
            # Multiplication factor for bootstrap current (note fbs~0.3 in
            # original simu)
            'bootstrap_mult': 1.0,
        },
        'generic_current': {
            # total "external" current fraction
            'fext': 0.46,
            # width of "external" Gaussian current profile (normalized radial
            # coordinate)
            'wext': 0.075,
            # radius of "external" Gaussian current profile (normalized radial
            # coordinate)
            'rext': 0.36,
        },
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {
            # total particle source
            'S_tot': 2.05e20,
            # particle source Gaussian central location (normalized radial
            # coordinate)
            'deposition_location': 0.3,
            # particle source Gaussian width (normalized radial coordinate)
            'particle_width': 0.25,
        },
        'gas_puff': {
            # pellets behave like a gas puff for this simulation with
            # exponential decay therefore use the puff structure for pellets
            # exponential decay length of gas puff ionization (normalized radial
            # coordinate)
            'puff_decay_length': 0.3,
            # total pellet particles/s
            'S_puff_tot': 6.0e21,
        },
        'pellet': {
            # total pellet particles/s (continuous pellet model)
            'S_pellet_tot': 0.0e22,
            # Gaussian width of pellet deposition (normalized radial coordinate)
            # in continuous pellet model
            'pellet_width': 0.1,
            # Pellet source Gaussian central location (normalized radial
            # coordinate) in continuous pellet model.
            'pellet_deposition_location': 0.85,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {
            'rsource': 0.12741589640723575,
            # Gaussian width in normalized radial coordinate r
            'w': 0.07280908366127758,
            # total heating (including accounting for radiation) r
            'Ptot': 51.0e6,
            # electron heating fraction r
            'el_heat_fraction': 0.68,
        },
        'fusion': {},
        'ei_exchange': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 1.0,
        },
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
        'Tiped': 4.5,  # ion pedestal top temperature in keV for T_i and T_e
        'Teped': (
            4.5
        ),  # electron pedestal top temperature in keV for T_i and T_e
        'neped': (
            0.62
        ),  # pedestal top electron density in units of density_reference
        'rho_norm_ped_top': 0.9,  # set ped top location in normalized radius
    },
    'transport': {
        'transport_model': 'qlknn',
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        'apply_inner_patch': True,
        'D_e_inner': 0.25,
        'V_e_inner': 0.0,
        'chi_i_inner': 1.0,
        'chi_e_inner': 1.0,
        'rho_inner': 0.2,  # radius below which patch transport is applied
        # allowed chi and diffusivity bounds
        'chi_min': 0.05,  # minimum chi
        'chi_max': 100,  # maximum chi (can be helpful for stability)
        'D_e_min': 0.05,  # minimum electron diffusivity
        # qlknn params.
        'DV_effective': True,
        'include_ITG': True,  # to toggle ITG modes on or off
        'include_TEM': True,  # to toggle TEM modes on or off
        'include_ETG': True,  # to toggle ETG modes on or off
        # ensure that smag - fusion > -0.2 always, to compensate for no slab
        # modes
        'avoid_big_negative_s': True,
        # minimum |R/Lne| below which effective V is used instead of
        # effective D
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        # (deliberately) large heat conductivity for Pereverzev rule
        'chi_pereverzev': 30,
        # (deliberately) large particle diffusion for Pereverzev rule
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
