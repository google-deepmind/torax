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

"""Tests qlknn particle transport with D_e from chi_e. CHEASE geometry.

Ip from parameters. current, heat, and particle transport. qlknn transport
model. Pedestal. Particle sources. PC method for density. D_e
scaled from chi_e
"""

CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,  # initial density (Greenwald fraction units)
        'n_e_bound_right': 0.2,
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
        't_final': 2,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'generic_heat': {
            'w': 0.18202270915319393,
        },
        'ei_exchange': {},
        'generic_particle': {
            'S_tot': 0.3e22,
        },
        'gas_puff': {
            'S_puff_tot': 0.5e22,
        },
        'pellet': {
            'S_pellet_tot': 1.0e22,
        },
        'j_bootstrap': {},
        'generic_current': {},
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
        'neped': 1.0,
    },
    'transport': {
        'transport_model': 'qlknn',
        # qlknn params.
        'DVeff': False,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
