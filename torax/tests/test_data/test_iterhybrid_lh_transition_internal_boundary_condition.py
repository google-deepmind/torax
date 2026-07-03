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

"""An ITER hybrid config with INTERNAL_BOUNDARY_CONDITION pedestal and LH transition.

Based on test_iterhybrid_lh_transition, but uses INTERNAL_BOUNDARY_CONDITION
mode with the formation model and a 3-second transition time, instead of
ADAPTIVE_TRANSPORT.
"""

import copy

from torax.tests.test_data import test_iterhybrid_lh_transition

CONFIG = copy.deepcopy(test_iterhybrid_lh_transition.CONFIG)

# Switch pedestal mode from ADAPTIVE_TRANSPORT to INTERNAL_BOUNDARY_CONDITION
# with formation model.
CONFIG['pedestal']['mode'] = 'INTERNAL_BOUNDARY_CONDITION'
CONFIG['pedestal'][
    'use_formation_model_with_internal_boundary_condition'
] = True
CONFIG['pedestal']['transition_time_width'] = 3.0

# INTERNAL_BOUNDARY_CONDITION does not need pedestal_transport_models since it
# uses source/sink terms rather than transport scaling.
del CONFIG['transport']['pedestal_transport_models']
