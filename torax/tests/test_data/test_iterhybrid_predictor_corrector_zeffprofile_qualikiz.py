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

"""Identical to test_iterhybrid_predictor_corrector_zeffprofile but with QuaLiKiz."""
import copy

from torax.tests.test_data import \
    test_iterhybrid_predictor_corrector_zeffprofile

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector_zeffprofile.CONFIG)


CONFIG['transport'] = {
    'model_name': 'qualikiz',
    # set inner core transport coefficients (ad-hoc MHD/EM transport)
    'apply_inner_patch': True,
    'D_e_inner': 0.25,
    'V_e_inner': 0.0,
    'chi_i_inner': 1.0,
    'chi_e_inner': 1.0,
    'rho_inner': 0.2,  # radius below which patch transport is applied
    # set outer core transport coefficients (L-mode near edge region)
    'apply_outer_patch': True,
    'D_e_outer': 0.1,
    'V_e_outer': 0.0,
    'chi_i_outer': 2.0,
    'chi_e_outer': 2.0,
    'rho_outer': 0.9,  # radius above which patch transport is applied
    # allowed chi and diffusivity bounds
    'chi_min': 0.05,  # minimum chi
    'chi_max': 100,  # maximum chi (can be helpful for stability)
    'D_e_min': 0.05,  # minimum electron diffusivity
    # qualikiz params.
    'DV_effective': True,
    'avoid_big_negative_s': True,
    'An_min': 0.05,
    'n_max_runs': 2,
    'n_processes': 24,
}
