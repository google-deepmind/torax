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

"""test_iterhybrid_predictor_corrector with Mavrin radiation, n_e_ratios, x-lengyel."""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

assert isinstance(CONFIG['plasma_composition'], dict)
# If present, Z_eff would be ignored in the 'n_e_ratios' impurity mode.
del CONFIG['plasma_composition']['Z_eff']
CONFIG['plasma_composition']['impurity'] = {
    'impurity_mode': 'n_e_ratios',
    'species': {'Ne': 0.01, 'W': 1e-5},
}
CONFIG['sources']['impurity_radiation'] = {
    'model_name': 'mavrin_fit',
}

CONFIG['edge'] = {
    'model_name': 'extended_lengyel',
    'computation_mode': 'forward',
    'fixed_impurity_concentrations': {'Ne': 0.01, 'W': 1e-5},
    'impurity_sot': 'core',
    'connection_length_target': 30.0,
    'connection_length_divertor': 10.0,
    'angle_of_incidence_target': 3.0,
    'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
    'toroidal_flux_expansion': 1.0,
    'use_enrichment_model': True,
    'diverted': True,
}
