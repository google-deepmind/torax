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
With Newton-Raphson solver and adaptive timestep (backtracking).
"""

CONFIG = {
    'plasma_composition': {
        # Bundled isotope average for ions
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'fractions',
            'species': 'Ne',
        },
        'Z_eff': 1.6,  # Sets impurity density
    },
    'profile_conditions': {
        'I_p': {0: 3e6, 80: 10.5e6},  # Total plasma current [A]
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'geometry',
        # Boundary and initial conditions for T and n
        'T_i': {0.0: {0.0: 6.0, 1.0: 0.1}},
        'T_i_right_bc': 0.1,
        'T_e': {0.0: {0.0: 6.0, 1.0: 0.1}},
        'T_e_right_bc': 0.1,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': {0: 0.1, 80: 0.3},
        # Set initial condition density according to Greenwald fraction
        'n_e_nbar_is_fGW': True,
        'nbar': 1,
        'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
    },
    'numerics': {
        # Shortens the current diffusion timescale (psi equation only)
        'current_diffusion_time_reduction_factor': 1,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
        'R_major': 6.2,  # Major radius R [m]
        'a_minor': 2.0,  # Minor radius a [m]
        'B_0': 5.3,  # Toroidal magnetic field on axis [T]
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {
            # Total "external" current fraction
            'fraction_of_total_current': 0.15,
            # Width and location of Gaussian profile (normalized radial
            # coordinate)
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },
        # Electron density sources/sinks (for the n_e equation)
        'generic_particle': {
            'S_total': 0.0,  # Total particle source [s^-1]
            # Location and width of Gaussian profile (normalized radial
            # coordinate)
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'gas_puff': {
            # Pellets behave like a gas puff with exponential decay for this
            # simulation, so use the puff structure for pellets
            # Exponential decay length of ionization (normalized radial
            # coordinate)
            'puff_decay_length': 0.3,
            'S_total': 0.0,  # Total pellet particles [s^-1]
        },
        'ohmic': {},
        'pellet': {
            # Continuous pellet model
            'S_total': 0.0e22,  # Total pellet particles [s^-1]
            # Width and location of Gaussian profile (normalized radial
            # coordinate)
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        # Ion and electron heat sources (for the T_i and T_e equations)
        'generic_heat': {
            # Location and width of Gaussian profile (normalized radial
            # coordinate)
            'gaussian_location': 0.12741589640723575,
            'gaussian_width': 0.07280908366127758,
            # Total heating (including accounting for radiation) [W]
            'P_total': 20.0e6,
            'electron_heat_fraction': 1.0,
        },
        'fusion': {},
        'ei_exchange': {
            # Multiplier for ion-electron heat exchange term for sensitivity
            'Qei_multiplier': 1.0,
        },
    },
    'pedestal': {
        'structure_model': {
            'model_name': 'set_T_ped_n_ped',
            'T_i_ped': 1.0,  # Ion pedestal top temperature [keV]
            'T_e_ped': 1.0,  # Electron pedestal top temperature [keV]
            'n_e_ped_is_fGW': True,
            'n_e_ped': {0: 0.3, 80: 0.7},  # Pedestal top electron density [fGW]
            'rho_norm_ped_top': 0.9,  # Pedestal top location
        },
    },
    'transport': {
        'core_transport_models': {
            'qlknn': {
                'model_name': 'qlknn',
                'DV_effective': True,
                'include_ITG': True,
                'include_TEM': True,
                'include_ETG': True,
                # Ensure that smag - alpha > -0.2 always, to compensate for
                # no slab modes
                'avoid_big_negative_s': True,
                # Minimum |R/Lne| below which effective V is used instead of
                # effective D
                'An_min': 0.05,
                'ITG_flux_ratio_correction': 1,
                'rho_min': 0.25,
                'rho_max': 0.95,
            },
            'inner_patch': {
                'model_name': 'prescribed',
                'chi_i': 1.5,
                'chi_e': 1.5,
                'D_e': 0.25,
                'V_e': 0.0,
                'rho_max': 0.25,
            },
            'outer_patch': {
                'model_name': 'prescribed',
                'chi_i': 2.0,
                'chi_e': 2.0,
                'D_e': 0.1,
                'V_e': 0.0,
                'rho_min': 0.95,
                'rho_max': 1.0,
            },
        },
        # Allowed chi and diffusivity bounds
        'chi_min': 0.05,
        'chi_max': 100,  # Can be helpful for stability
        'D_e_min': 0.05,
        'V_e_max': 10,
        'smoothing_width': 0.1,
    },
    'solver': {
        'solver_type': 'newton_raphson',
        # TODO(b/515255142): Improve the warm-start predictor-corrector so that
        # it can be re-enabled.
        'initial_guess_mode': 'x_old',
        'use_fixed_point': True,
        'n_fixed_point_iterations': 5,
        # (Deliberately) large heat conductivity and particle diffusion for
        # Pereverzev rule
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        # use_pereverzev is only used for the linear solver
        'use_pereverzev': True,
        'vmap_linesearch': True,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
        'dt': 2,
        'dt_backtracking_factor': 3,
        't_final': 80,  # Simulated plasma duration [s]
    },
}
