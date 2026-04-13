# Copyright 2026 DeepMind Technologies Limited
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

"""An ITER hybrid config with ADAPTIVE_TRANSPORT pedestal and LH transition.

This configuration is designed to test the L-mode to H-mode transition.
Low heating for t=0 to 30, ramp to reach high heating at t=60.

- Pedestal formation governed by Martin scaling.
- Pedestal height from user.
- Nonlinear solver.
- Evolve Te, Ti, psi only.
"""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector


CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

# Change initial conditions to L-mode
CONFIG['profile_conditions']['T_i'] = {
    0.0: {0.0: 3.0, 1.0: 0.25},
}
CONFIG['profile_conditions']['T_e'] = {
    0.0: {0.0: 3.0, 1.0: 0.25},
}
CONFIG['profile_conditions']['n_e'] = {
    0.0: {0.0: 0.5e20, 1.0: 0.15e20},
}
CONFIG['profile_conditions']['n_e_right_bc'] = 0.15e20
CONFIG['profile_conditions']['n_e_right_bc_is_fGW'] = False
CONFIG['profile_conditions']['n_e_nbar_is_fGW'] = False
CONFIG['profile_conditions']['normalize_n_e_to_nbar'] = False

# Low heating for t=0 to 10, ramp to reach high heating at t=50.
CONFIG['numerics']['t_initial'] = 0
CONFIG['numerics']['t_final'] = 60
CONFIG['sources']['generic_heat']['P_total'] = {
    0.0: 1e6,
    10.0: 1e6,
    50.0: 6e7,
}

# Set transport
CONFIG['transport'] = {
    'model_name': 'combined',
    'transport_models': [
        {
            'model_name': 'constant',
            'rho_max': 0.2,
            'chi_i': 1.0,
            'chi_e': 1.0,
            'D_e': 0.25,
            'V_e': 0.0,
        },
        {
            'model_name': 'qlknn',
            'rho_min': 0.2,
            'rho_max': 0.8,
        },
        # Before the pedestal forms, we set the transport to a constant value
        # in the edge region for stable L-mode operation.
        {
            'model_name': 'constant',
            'rho_min': 0.8,
            'chi_i': 1.0,
            'chi_e': 1.0,
            'D_e': 1e-3,
            'V_e': 0.0,
        },
    ],
    'pedestal_transport_models': [
        # TODO:
        # The final desired behavior is for the contributions from the core
        # transport models to be scaled according to the ADAPTIVE_TRANSPORT
        # mode, and then the contributions from the pedestal transport models
        # added to these afterwards. This is not yet implemented.
        # Currently, the combined transport model completely masks out
        # contributions from the core transport models in the pedestal region,
        # leaving nothing to scale with the ADAPTIVE_TRANSPORT mode.
        # We therefore provide a duplicate of the core transport for rho > 0.8
        # here, which will be scaled correctly.
        # This case has the following behaviour:
        # - If P_sol < P_LH, transport is
        #   [constant, qlknn, constant]
        # - If P_sol > P_LH, transport is
        #   [constant, qlknn, constant*ADAPTIVE_TRANSPORT_multiplier]
        {
            'model_name': 'constant',
            'chi_i': 1.0,
            'chi_e': 1.0,
            'D_e': 1e-3,
            'V_e': 0.0,
        },
    ],
}

# Set pedestal to adaptive transport mode
CONFIG['pedestal'] = {
    'model_name': 'set_T_ped_n_ped',
    'set_pedestal': True,
    'mode': 'ADAPTIVE_TRANSPORT',
    'T_i_ped': 4.5,
    'T_e_ped': 4.5,
    'n_e_ped': 0.62e20,
    'rho_norm_ped_top': 0.9,
    'formation_model': {
        'model_name': 'martin_scaling',
        'sharpness': 10.0,
    },
    'saturation_model': {
        'model_name': 'profile_value',
        'steepness': 100.0,
    },
    'pedestal_top_smoothing_width': 0.02,
}

# Use nonlinear solver, as linear solver struggles with the fast dynamics of the
# L-H transition.
CONFIG['solver'] = {
    'solver_type': 'newton_raphson',
    'use_predictor_corrector': True,
    'use_pereverzev': True,
}

# Nonlinear solver requires more smoothness in the transport coefficients.
CONFIG['transport']['smoothing_width'] = 0.1

# Use the from_previous_dt time step calculator to speed up the simulation.
CONFIG['time_step_calculator'] = {'calculator_type': 'from_previous_dt'}
CONFIG['numerics']['min_dt'] = 1e-3
CONFIG['numerics']['max_dt'] = 1.0
CONFIG['numerics']['fixed_dt'] = 1.0
CONFIG['numerics']['dt_reduction_factor'] = 2.0

# Turn off resistivity multiplier, as otherwise high chi -> low temp -> high
# resistivity becomes unstable.
CONFIG['numerics']['resistivity_multiplier'] = 1.0

# Disable density evolution. Tuning the density for the LH transition is
# quite hard as it involves the edge boundary condition as well as the pellet
# and gas puff sources.
CONFIG['numerics']['evolve_density'] = False
