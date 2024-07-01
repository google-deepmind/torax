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

"""test_fusion_power: with current, heat, and particle transport.

CGM transport model. Pedestal. All particle sources. Self-consistent
fusion power.
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': True,
            'nbar': 0.85,  # initial density (in Greenwald fraction units)
            'ne_bound_right': 0.2,
            'neped': 1.0,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            'nu': 0,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 1,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
        },
        'jext': {},
        # Electron density sources/sink (for the ne equation).
        'nbi_particle_source': {
            # NBI total particle source
            'S_nbi_tot': 0.3e22,
        },
        'gas_puff_source': {
            # total pellet particles/s
            'S_puff_tot': 0.5e22,
        },
        'pellet_source': {
            # total pellet particles/s (continuous pellet model)
            'S_pellet_tot': 1.0e22,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {
            # total heating (including accounting for radiation)
            'Ptot': 53.0e6,
        },
        'fusion_heat_source': {},
        'qei_source': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 1.0,
        },
    },
    'transport': {
        'transport_model': 'CGM',
        'cgm_params': {
            # CGM model ratio of ion heat conductivity to particle diffusion
            'chi_D_ratio': 8,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
        'use_pereverzev': True,
        'd_per': 0.0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
