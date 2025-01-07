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

Ip from parameters. implicit + pereverzev-corrigan, Ti+Te+Psi, Pei standard
dens, pedestal, chi from QLKNN. Includes time dependent Ip, Ptot, and
pedestal, mocking up current-overshoot and an LH transition
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'Ti': {0.0: {0.0: 10.0, 1.0: 1.0}},
            'Te': {0.0: {0.0: 10.0, 1.0: 1.0}},
            'Ip_tot': {0: 5, 4: 15, 6: 12, 8: 12},
            'ne_bound_right': 0.5,
        },
        'numerics': {
            'current_eq': True,
            # to shorten current diffusion time for the test
            'resistivity_mult': 50,
            'dtmult': 150,
            'maxdt': 0.5,
            't_final': 10,
            'enable_prescribed_profile_evolution': False,
        },
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_ion_el_heat_source': {
            'w': 0.18202270915319393,
            'Ptot': {
                0: 20e6,
                9: 20e6,
                10: 120e6,
                15: 120e6,
            },  # in W
        },
        'qei_source': {},
        'generic_current_source': {},
    },
    'pedestal': {
        'Tiped': {0: 2, 4: 2, 6: 5, 8: 4},
        'Teped': {0: 2, 4: 2, 6: 5, 8: 4},
    },
    'transport': {
        'transport_model': 'qlknn',
        'apply_inner_patch': True,
        'chii_inner': 2.0,
        'chie_inner': 2.0,
        'rho_inner': 0.3,
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
