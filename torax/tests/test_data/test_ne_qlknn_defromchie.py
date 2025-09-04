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
        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,
        'nbar': 0.85,  # initial density (Greenwald fraction units)
        'n_e_right_bc': 0.2e20,
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
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'generic_heat': {
            'gaussian_width': 0.18202270915319393,
        },
        'ei_exchange': {},
        'generic_particle': {
            'S_total': 0.3e22,
        },
        'gas_puff': {
            'S_total': 0.5e22,
        },
        'pellet': {
            'S_total': 1.0e22,
        },
        'generic_current': {},
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'n_e_ped': 1.0e20,
    },
    'transport': {
        'model_name': 'qlknn',
        # qlknn params.
        'DV_effective': False,
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
