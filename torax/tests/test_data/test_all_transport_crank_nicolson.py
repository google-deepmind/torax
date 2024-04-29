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

"""Tests theta=0.5 (Crank-Nicolson) with all transport equations.

Ip from parameters. current, heat, and particle transport. qlknn transport
model. Pedestal. Particle sources including NBI. PC method for density. Deff +
Veff model.
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': True,
            'nbar': 0.85,  # initial density (using Greenwald fraction default)
            'ne_bound_right': 0.2,
            'neped': 1.0,
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
            'largeValue_n': 1.0e5,
        },
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        # Current sources (for psi equation)
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
        },
        'jext': {},
        # Electron density sources/sink (for the ne equation).
        'nbi_particle_source': {
            'S_nbi_tot': 0.3e22,
        },
        'gas_puff_source': {
            'S_puff_tot': 0.5e22,
        },
        'pellet_source': {
            'S_pellet_tot': 1.0e22,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {
            'w': 0.18202270915319393,
        },
        'qei_source': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 1.0,
        },
    },
    'transport': {
        'transport_model': 'qlknn',
        'qlknn_params': {
            'DVeff': True,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
        'use_pereverzev': True,
        'chi_per': 60.0,
        'd_per': 30.0,
        'theta_imp': 0.5,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
