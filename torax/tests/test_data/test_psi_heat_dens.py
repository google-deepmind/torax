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

"""Tests combined current, heat, and particle transport, with a pedestal.

Constant transport coefficient model. Pedestal
"""

CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,  # initial density (in Greenwald fraction units)
        'n_e_right_bc': 0.5,
        'nu': 0,
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
        'generic_heat': {},
        'ei_exchange': {},
        'j_bootstrap': {},
        'generic_current': {},
    },
    'pedestal': {'pedestal_model': 'set_tped_nped', 'set_pedestal': True},
    'transport': {
        'transport_model': 'constant',
        # constant params.
        # diffusion coefficient in electron density equation in m^2/s
        'D_e': 0.5,
        # convection coefficient in electron density equation in m^2/s
        'V_e': -0.2,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
