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

"""Tests Bremsstrahlung power model with time-dependent Zimp and Zeff. CHEASE.

Ip from parameters. implicit, Ti+Te, Pei low dens, no pedestal, constant chi
"""


CONFIG = {
    'runtime_params': {
        'plasma_composition': {
            'Zeff': {0: {0: 3.0, 1: 3.0}, 1: {0: 5.0, 1: 5.0}},
            'Zimp': {0: 10.0, 1: 30.0},
        },
        'profile_conditions': {
            'set_pedestal': True,
            'ne_is_fGW': True,
            'ne_bound_right': 0.5,
            'nbar': 0.95,  # initial density (Greenwald fraction units)
        },
        'numerics': {
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 1,
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
        },
        'qei_source': {},
        'ohmic_heat_source': {},
        'generic_current_source': {},
        'bremsstrahlung_heat_sink': {
            'use_relativistic_correction': True,
        },
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            'chii_const': 0.5,
            'chie_const': 0.5,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
