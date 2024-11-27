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


# Define the generic_current_source profile
def gaussian(r, center, width, amplitude):
  return amplitude * np.exp(-((r - center) ** 2) / (2 * width**2))


times = np.array([0, 2.5])
gauss_r = np.linspace(0, 1, 32)
generic_current_source_profiles = np.array([
    gaussian(gauss_r, center=0.35, width=0.05, amplitude=1e6),
    gaussian(gauss_r, center=0.15, width=0.1, amplitude=1e6),
])

# Create the config
CONFIG = {
    'pedestal': {},
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': False,
            'nbar': 0.85,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            # 'nu': 0,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 5,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Only drive the external current source
        'generic_current_source': {
            'mode': 'PRESCRIBED',
            'prescribed_values': (
                times,
                gauss_r,
                generic_current_source_profiles,
            ),
        },
        # Disable density sources/sinks
        'generic_particle_source': {
            'S_tot': 0.0,
        },
        'gas_puff_source': {
            'S_puff_tot': 0.0,
        },
        'pellet_source': {
            'S_pellet_tot': 0.0,
        },
        # Use default heat sources
        'generic_ion_el_heat_source': {},
        'qei_source': {},
    },
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            # diffusion coefficient in electron density equation in m^2/s
            'De_const': 0.5,
            # convection coefficient in electron density equation in m^2/s
            'Ve_const': -0.2,
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
