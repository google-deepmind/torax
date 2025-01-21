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

"""Config for ITER hybrid scenario based parameters with nonlinear solver.

ITER hybrid scenario based (roughly) on van Mulders Nucl. Fusion 2021.
With Newton-Raphson stepper and adaptive timestep (backtracking)
"""


CONFIG = {
    'runtime_params': {
        'plasma_composition': {
            'main_ion': {'D': 0.5, 'T': 0.5},  # (bundled isotope average)
            'impurity': 'Ne',
            'Zeff': 1.6,  # sets impurity density
        },
        'profile_conditions': {
            'Ip_tot': {0: 3, 80: 10.5},  # total plasma current in MA
            # boundary + initial conditions for T and n
            # initial condition ion temperature for r=0 and r=Rmin
            'Ti': {0.0: {0.0: 6.0, 1.0: 0.1}},
            'Ti_bound_right': (
                0.1
            ),  # boundary condition ion temperature for r=Rmin
            # initial condition electron temperature for r=0 and r=Rmin
            'Te': {0.0: {0.0: 6.0, 1.0: 0.1}},
            'Te_bound_right': (
                0.1
            ),  # boundary condition electron temp for r=Rmin
            'ne_bound_right_is_fGW': True,
            # boundary condition density for r=Rmin
            'ne_bound_right': {0: 0.1, 80: 0.3},
            # set initial condition density according to Greenwald fraction.
            'ne_is_fGW': True,
            'nbar': 1,
            'ne': {0: {0.0: 1.5, 1.0: 1.0}},  # Initial electron density profile
            # internal boundary condition (pedestal)
            # do not set internal boundary condition if this is False
            'set_pedestal': True,
        },
        'numerics': {
            # simulation control
            't_final': 80,  # length of simulation time in seconds
            'fixed_dt': 2,
            # 1/multiplication factor for sigma (conductivity) to reduce current
            # diffusion timescale to be closer to heat diffusion timescale.
            'resistivity_mult': 1,
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'current_eq': True,
            'dens_eq': True,
            'maxdt': 0.5,
            # multiplier in front of the base timestep dt=dx^2/(2*chi). Can
            # likely be increased further beyond this default.
            'dtmult': 30,
            'dt_reduction_factor': 3,
        },
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
            'bootstrap_mult': 1.0,
        },
        'generic_current_source': {
            # total "external" current fraction
            'fext': 0.15,
            # width of "external" Gaussian current profile (normalized radial
            # coordinate)
            'wext': 0.075,
            # radius of "external" Gaussian current profile (normalized radial
            # coordinate)
            'rext': 0.36,
        },
        # Electron density sources/sink (for the ne equation).
        'generic_particle_source': {
            # total particle source
            'S_tot': 0.0,
            # particle source Gaussian central location (normalized radial
            # coordinate)
            'deposition_location': 0.3,
            # particle source Gaussian width (normalized radial coordinate)
            'particle_width': 0.25,
        },
        'gas_puff_source': {
            # pellets behave like a gas puff for this simulation with
            # exponential decay therefore use the puff structure for pellets
            # exponential decay length of gas puff ionization (normalized radial
            # coordinate)
            'puff_decay_length': 0.3,
            # total pellet particles/s
            'S_puff_tot': 0.0,
        },
        'pellet_source': {
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
        'generic_ion_el_heat_source': {
            'rsource': 0.12741589640723575,
            # Gaussian width in normalized radial coordinate r
            'w': 0.07280908366127758,
            # total heating (including accounting for radiation) r
            'Ptot': 20.0e6,
            # electron heating fraction r
            'el_heat_fraction': 1.0,
        },
        'fusion_heat_source': {},
        'qei_source': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 1.0,
        },
    },
    'pedestal': {
        'Tiped': 1.0,  # ion pedestal top temperature in keV for Ti and Te
        'Teped': 1.0,  # electron pedestal top temperature in keV for Ti and Te
        'neped_is_fGW': True,
        # pedestal top electron density in units of nref
        'neped': {0: 0.3, 80: 0.7},
        'rho_norm_ped_top': 0.9,  # set ped top location in normalized radius
    },
    'transport': {
        'transport_model': 'qlknn',
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        'apply_inner_patch': True,
        'De_inner': 0.25,
        'Ve_inner': 0.0,
        'chii_inner': 1.5,
        'chie_inner': 1.5,
        'rho_inner': 0.3,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        'apply_outer_patch': True,
        'De_outer': 0.1,
        'Ve_outer': 0.0,
        'chii_outer': 2.0,
        'chie_outer': 2.0,
        'rho_outer': 0.9,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        'chimin': 0.05,  # minimum chi
        'chimax': 100,  # maximum chi (can be helpful for stability)
        'Demin': 0.05,  # minimum electron diffusivity
        'Demax': 50,  # maximum electron diffusivity
        'Vemin': -10,  # minimum electron convection
        'Vemax': 10,  # minimum electron convection
        'smoothing_sigma': 0.1,
        'qlknn_params': {
            'DVeff': True,
            'include_ITG': True,  # to toggle ITG modes on or off
            'include_TEM': True,  # to toggle TEM modes on or off
            'include_ETG': True,  # to toggle ETG modes on or off
            # ensure that smag - alpha > -0.2 always, to compensate for no slab
            # modes
            'avoid_big_negative_s': True,
            # minimum |R/Lne| below which effective V is used instead of
            # effective D
            'An_min': 0.05,
            'ITG_flux_ratio_correction': 1,
        },
    },
    'stepper': {
        'stepper_type': 'newton_raphson',
        'predictor_corrector': True,
        'corrector_steps': 10,
        # (deliberately) large heat conductivity for Pereverzev rule
        'chi_per': 30,
        # (deliberately) large particle diffusion for Pereverzev rule
        'd_per': 15,
        # use_pereverzev is only used for the linear solver
        'use_pereverzev': True,
        'newton_raphson_params': {
            'log_iterations': True,
        },
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
