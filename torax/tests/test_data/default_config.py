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

"""Run with the default general_runtime_params."""


# Note, for backwards-compatibility, the "default" test in this case has many
# sources turned on as well.


CONFIG = {
    'profile_conditions': {},
    'plasma_composition': {},
    'numerics': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {'bootstrap_current': {}},
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
