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

"""Self-contained example of the HPI2-NN pellet source.

The purpose of this example is simply to test that the coupling between 
torax and hpi2nn is working properly and to show a minimal configuration.

Requirements:
  - The 'hpi2nn' package must be installed (pip install -e <hpi2nn repo>), it
    provides the surrogate model and its weights. See docs/hpi2nn_pellet_source.
"""

CONFIG = {
    'profile_conditions': {},  # default profile conditions
    'plasma_composition': {},  # default plasma composition
    'numerics': {
        't_initial': 0.0,
        't_final': 10,
        'fixed_dt': 0.01,
        'min_dt': 1e-4,
        # The density equation must be evolved for the pellet to fuel the core.
        'evolve_density': True,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
    },
    # Circular geometry is only for testing and prototyping (no external files).
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        # Current source (for the psi equation).
        'generic_current': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'ei_exchange': {},
        'ohmic': {},
        # HPI2-NN pellet particle source (for the n_e equation).
        'hpi2nn_pellet_source': {
            'trigger_times': [2.0, 6.0],
            'pellet_radius': 1.0e-3,  # [m]
            'pellet_velocity': 100.0,  # [m/s]
            'injection_line': 'WEST_upHFS',
            'use_hpi2nn_ablation_time': False,
            'ablation_time': 1e-3,  # [s]
        },
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },

    'solver': {
        'solver_type': 'linear',
    },
    # Mandatory for the HPI2-NN pellet source.
    'time_step_calculator': {
        'calculator_type': 'pellet_aware',
        'base_calculator_type': 'fixed',
    },
}
