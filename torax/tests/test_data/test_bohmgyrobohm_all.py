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

"""Tests BgB transport.

- BgB transport model (heat + particle transport)
- Linear stepper with Pereverzev-Corrigan
- Chi time step calculator
- Circular geometry
- Sources:
  - No bootstrap
  - generic_current
  - generic particle source
  - gas puff
  - pellet
  - generic ion-el heat source
  - qei source
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'nbar': 0.8,
            'ne_bound_right': 0.5,
            'ne_bound_right_is_fGW': True,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 2,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        'j_bootstrap': {},
        # Electron density sources/sink (for the ne equation).
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'ei_exchange': {},
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
        'neped': 0.8,
        'neped_is_fGW': True,
    },
    'transport': {
        'transport_model': 'bohm-gyrobohm',
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
