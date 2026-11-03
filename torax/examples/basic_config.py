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

"""Simplified config using mostly defaults for various simulation components."""


CONFIG = {
    'profile_conditions': {
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
    },  # use default profile conditions
    'plasma_composition': {},  # use default plasma composition
    'numerics': {},  # use default numerics
    # circular geometry is only for testing and prototyping
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {},
        'gas_puff': {
            'S_total': 1e22,
            'puff_decay_length': 0.05,
        },
        'pellet': {
            'S_total': 2e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {},
    'transport': {
        'core_transport_models': {
            'prescribed': {'model_name': 'prescribed'},
        },
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
