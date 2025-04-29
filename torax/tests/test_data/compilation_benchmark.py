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

"""Compilating time benchmark script.

This config is not run as an actual automated test, but is convenient to have
as a manual test to invoke from time to time. It is configured to incur high
but not extreme compile time: high enough that compile time problems are
obvious, not so high that iterating on solutions to compile time problems
is infeasible.
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'nbar': 0.85,  # initial density (Greenwald fraction units)
            'ne_bound_right': 0.2,
            'neped': 1.0,
            'nu': 0,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 0.0007944 * 2,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'ei_exchange': {
            'Qei_mult': 1.0,
        },
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
        },
        'generic_heat': {
            'Ptot': 53.0e6,
        },
        'pellet': {
            'S_pellet_tot': 1.0e22,
        },
        'gas_puff': {
            'S_puff_tot': 0.5e22,
        },
        'generic_particle': {
            'S_tot': 0.3e22,
        },
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
    },
    'transport': {
        'transport_model': 'qlknn',
        'qlknn_params': {
            'DVeff': False,
        },
    },
    'stepper': {
        'stepper_type': 'newton_raphson',
        'use_pereverzev': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
