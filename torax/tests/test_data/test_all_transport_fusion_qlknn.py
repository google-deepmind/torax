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

"""Tests combined current, heat, and particle transport with QLKNN.

qlknn transport model. Pedestal. Particle sources. PC method for
density. D_e scaled from chi_e
"""


CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,  # initial density (Greenwald fraction units)
        'n_e_right_bc': 0.2,
        # set flat Ohmic current to provide larger range of current
        # evolution for test
        'current_profile_nu': 0,
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
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
        },
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {
            'S_tot': 0.3e22,
        },
        'gas_puff': {
            'S_puff_tot': 0.5e22,
        },
        'pellet': {
            'S_pellet_tot': 1.0e22,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {
            'Ptot': 53.0e6,
        },
        'fusion': {},
        'ei_exchange': {
            'Qei_mult': 1.0,
        },
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
        'neped': 1.0,
    },
    'transport': {
        'transport_model': 'qlknn',
        'DV_effective': False,
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
