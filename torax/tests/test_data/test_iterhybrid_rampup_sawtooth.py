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

"""Similar to test_iterhybrid_rampup but with sawteeth."""
import copy
from torax.tests.test_data import test_iterhybrid_rampup


CONFIG = copy.deepcopy(test_iterhybrid_rampup.CONFIG)
assert(isinstance(CONFIG['profile_conditions'], dict))
CONFIG['profile_conditions']['initial_psi_from_j'] = True
CONFIG['profile_conditions']['nu'] = 2.0
CONFIG['sources']['generic_current']['fext'] = 0.0
CONFIG['sources']['generic_heat']['Ptot'] = 0.0
CONFIG['sources']['ohmic'] = {}
CONFIG['numerics']['resistivity_multiplier'] = 10.0
CONFIG['numerics']['fixed_dt'] = 1.0
CONFIG['transport']['chi_i_inner'] = 0.2
CONFIG['transport']['chi_e_inner'] = 0.2
CONFIG['transport']['D_e_inner'] = 0.02
CONFIG['transport']['rho_inner'] = 0.15
CONFIG['mhd'] = {
    'sawtooth': {
        'trigger_model': {
            'trigger_model_type': 'simple',
            'minimum_radius': 0.05,
            's_critical': 0.15,
        },
        'redistribution_model': {
            'redistribution_model_type': 'simple',
            'flattening_factor': 1.001,
            'mixing_radius_multiplier': 1.3,
        },
    }
}
