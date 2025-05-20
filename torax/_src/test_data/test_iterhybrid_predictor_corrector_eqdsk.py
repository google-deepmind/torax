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

"""Identical to test_iterhybrid_predictor_corrector but EQDSK geometry input."""
import copy
from torax.tests.test_data import test_iterhybrid_predictor_corrector


CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG['geometry'] = {
    'geometry_type': 'eqdsk',
    'geometry_file': 'EQDSK_ITERhybrid_COCOS02.eqdsk',
    'Ip_from_parameters': True,
    'last_surface_factor': 0.99,
    'n_surfaces': 100,
}
