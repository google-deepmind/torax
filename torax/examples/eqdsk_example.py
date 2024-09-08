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

"""ITER hybrid scenario based (roughly) on van Mulders NF 2021."""


CONFIG = {
    'runtime_params': {},
    'geometry': {
        'geometry_type': 'EQDSK',
        'geometry_file': 'eqdsk_cocos02.eqdsk',
        'Ip_from_parameters': True,
    },
    'sources': {
        # Current sources (for psi equation)
        'j_bootstrap': {},
        'jext': {},
        # Electron density sources/sink (for the ne equation).
        'nbi_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {},
        'fusion_heat_source': {},
        'qei_source': {},
        'ohmic_heat_source': {},
    },
    'transport': {
        'transport_model': 'qlknn',
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        'apply_inner_patch': True,
        'De_inner': 0.25,
        'Ve_inner': 0.0,
        'chii_inner': 1.0,
        'chie_inner': 1.0,
        'rho_inner': 0.2,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        'apply_outer_patch': True,
        'De_outer': 0.1,
        'Ve_outer': 0.0,
        'chii_outer': 2.0,
        'chie_outer': 2.0,
        'rho_outer': 0.9,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        'chimin': 0.05,  # minimum chi
        'chimax': 100,  # maximum chi (can be helpful for stability)
        'Demin': 0.05,  # minimum electron diffusivity
        'qlknn_params': {
            'DVeff': True,
            'coll_mult': 0.25,
            'include_ITG': True,  # to toggle ITG modes on or off
            'include_TEM': True,  # to toggle TEM modes on or off
            'include_ETG': True,  # to toggle ETG modes on or off
            # ensure that smag - alpha > -0.2 always, to compensate for no slab
            # modes
            'avoid_big_negative_s': True,
            # minimum |R/Lne| below which effective V is used instead of
            # effective D
            'An_min': 0.05,
            'ITG_flux_ratio_correction': 1,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': True,
        'corrector_steps': 1,
        # (deliberately) large heat conductivity for Pereverzev rule
        'chi_per': 30,
        # (deliberately) large particle diffusion for Pereverzev rule
        'd_per': 15,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
