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
    'profile_conditions': {
        'nbar': 0.8,
        'n_e_right_bc': 0.5,
        'n_e_right_bc_is_fGW': True,
    },
    'numerics': {
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
        'evolve_current': True,
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 2,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        'j_bootstrap': {},
        # Electron density sources/sink (for the n_e equation).
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
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
