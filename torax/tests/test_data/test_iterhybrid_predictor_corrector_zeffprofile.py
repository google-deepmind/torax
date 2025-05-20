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

"""Identical to test_iterhybrid_predictor_corrector but with Z_eff profile."""
import copy
import numpy as np
from torax.tests.test_data import test_iterhybrid_predictor_corrector


CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

# Define zeff profile, parabolic with min and max values.
rhon = np.linspace(0, 1, 50)
zeff_min = 1.2
zeff_max = 2.8
zeff_diff = zeff_max - zeff_min
zeff = -4 * zeff_diff * rhon**2 + 4 * zeff_diff * rhon + zeff_min

CONFIG['plasma_composition']['Z_eff'] = (rhon, zeff)
