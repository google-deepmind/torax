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
    'runtime_params': {},
    'geometry': {
        'geometry_type': 'circular',
        'nr': 25,
        'geometry_configs': {
            0: {
                'kappa': 1.,
            },
            5: {
                'kappa': 1.7,
            }
        },
    },
    'sources': {
        # Current sources (for psi equation)
        'j_bootstrap': {},
        'jext': {},
        # Electron density sources/sink (for the ne equation).
        'nbi_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {},
        'fusion_heat_source': {},
        'qei_source': {},
        'ohmic_heat_source': {},
    },
    'transport': {
        'transport_model': 'constant',
    },
    'stepper': {
        'stepper_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
