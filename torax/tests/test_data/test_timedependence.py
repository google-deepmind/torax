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

"""Tests time dependent boundary conditions and sources.

Ip from parameters. implicit + pereverzev-corrigan, T_i+T_e+Psi, Pei standard
dens, pedestal, chi from QLKNN. Includes time dependent Ip, Ptot, and
pedestal, mocking up current-overshoot and an LH transition
"""

CONFIG = {
    'profile_conditions': {
        'T_i': {0.0: {0.0: 10.0, 1.0: 1.0}},
        'T_e': {0.0: {0.0: 10.0, 1.0: 1.0}},
        'Ip_tot': {0: 5, 4: 15, 6: 12, 8: 12},
        'n_e_right_bc': 0.5,
        'n_e_nbar_is_fGW': False,
        'nbar': 0.33820425407,  # fGW=0.85 for Ip_tot(t=0)
    },
    'numerics': {
        'evolve_current': True,
        # to shorten current diffusion time for the test
        'resistivity_multiplier': 50,
        'chi_timestep_prefactor': 150,
        'max_dt': 0.5,
        't_final': 10,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_heat': {
            'w': 0.18202270915319393,
            'Ptot': {
                0: 20e6,
                9: 20e6,
                10: 120e6,
                15: 120e6,
            },  # in W
        },
        'ei_exchange': {},
        'generic_current': {},
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
        'Tiped': {0: 2, 4: 2, 6: 5, 8: 4},
        'Teped': {0: 2, 4: 2, 6: 5, 8: 4},
    },
    'transport': {
        'transport_model': 'qlknn',
        'apply_inner_patch': True,
        'chi_i_inner': 2.0,
        'chi_e_inner': 2.0,
        'rho_inner': 0.3,
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
