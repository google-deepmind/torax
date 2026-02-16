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

"""Tests internal boundary conditions."""

import copy
from torax.tests.test_data import test_timedependence

CONFIG = copy.deepcopy(test_timedependence.CONFIG)
CONFIG['profile_conditions']['internal_boundary_conditions'] = {
    'T_e': {0.0: {0.4: 7.0}, 2.0: {0.4: 10.0}, 5.0: {0.4: 15.0}},
}
