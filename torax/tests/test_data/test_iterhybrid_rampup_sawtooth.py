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
assert(isinstance(CONFIG['runtime_params']['profile_conditions'], dict))
CONFIG['runtime_params']['profile_conditions']['initial_psi_from_j'] = True
CONFIG['runtime_params']['profile_conditions']['nu'] = 2.0
CONFIG['sources']['generic_current_source']['fext'] = 0.0
CONFIG['sources']['generic_ion_el_heat_source']['Ptot'] = 0.0
CONFIG['sources']['ohmic_heat_source'] = {}
CONFIG['runtime_params']['numerics']['resistivity_mult'] = 10.0
CONFIG['runtime_params']['numerics']['fixed_dt'] = 1.0
CONFIG['transport']['chii_inner'] = 0.2
CONFIG['transport']['chie_inner'] = 0.2
CONFIG['transport']['De_inner'] = 0.02
CONFIG['transport']['rho_inner'] = 0.15
CONFIG['mhd'] = {
    'sawtooth': {
        'trigger_model_config': {
            'trigger_model_type': 'simple',
            'minimum_radius': 0.05,
            's_critical': 0.15,
        },
        'redistribution_model_config': {
            'redistribution_model_type': 'simple',
            'flattening_factor': 1.001,
            'mixing_radius_multiplier': 1.3,
        },
    }
}
