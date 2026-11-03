# Copyright 2026 DeepMind Technologies Limited
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

"""Similar to test_iterhybrid_rampup with IMAS prescribed psi from geometry."""
import copy

from torax.tests.test_data import test_iterhybrid_rampup

CONFIG = copy.deepcopy(test_iterhybrid_rampup.CONFIG)
assert isinstance(CONFIG['profile_conditions'], dict)
assert isinstance(CONFIG['numerics'], dict)
CONFIG['geometry'] = {
    'geometry_type': 'imas',
    'imas_filepath': 'ITERhybrid_rampup_11_time_slices_COCOS17_IDS_ddv4.nc',
}
del CONFIG['profile_conditions']['I_p']
CONFIG['profile_conditions']['I_p_source'] = 'geometry'
CONFIG['profile_conditions']['psi_source'] = 'geometry'
CONFIG['numerics']['evolve_current'] = False
