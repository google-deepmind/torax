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


"""Tests CHEASE geometry when setting Ip from config, with varying vloop.

* Ip from parameters
* implicit
* psi (current diffusion) only
* varying v loop boundary condition
"""

import numpy as np

times = np.linspace(0, 3, 100)
# 1 Hz frequency
vloop = 8.7 + 10 * np.sin(2 * np.pi * times)


CONFIG = {
    'profile_conditions': {
        'ne_bound_right': 0.5,
        'use_vloop_lcfs_boundary_condition': True,
        'vloop_lcfs': (times, vloop),
    },
    'numerics': {
        'ion_heat_eq': False,
        'el_heat_eq': False,
        'current_eq': True,
        'resistivity_mult': 100,  # to shorten current diffusion time
        't_final': 3,
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
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
    },
    'stepper': {
        'solver_type': 'linear',
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
