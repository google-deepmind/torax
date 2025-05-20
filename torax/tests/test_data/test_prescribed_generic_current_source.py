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

"""Tests driving a prescribed, time-varying external current source.

Constant transport coefficient model, circular geometry.
"""
import numpy as np


# Define the generic_current profile
def gaussian(r, center, width, amplitude):
  return amplitude * np.exp(-((r - center) ** 2) / (2 * width**2))


times = np.array([0, 2.5])
gauss_r = np.linspace(0, 1, 32)
generic_current_profiles = np.array([
    gaussian(gauss_r, center=0.35, width=0.05, amplitude=1e6),
    gaussian(gauss_r, center=0.15, width=0.1, amplitude=1e6),
])

# Create the config
CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,
        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,
        # set flat Ohmic current to provide larger range of current
        # evolution for test
        # 'current_profile_nu': 0,
    },
    'numerics': {
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
        'evolve_current': True,
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 5,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Only drive the external current source
        'generic_current': {
            'mode': 'PRESCRIBED',
            'prescribed_values': (
                (
                    times,
                    gauss_r,
                    generic_current_profiles,
                ),
            ),
        },
        # Disable density sources/sinks
        'generic_particle': {
            'S_total': 0.0,
        },
        'gas_puff': {
            'S_total': 0.0,
        },
        'pellet': {
            'S_total': 0.0,
        },
        # Use default heat sources
        'generic_heat': {},
        'ei_exchange': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
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
