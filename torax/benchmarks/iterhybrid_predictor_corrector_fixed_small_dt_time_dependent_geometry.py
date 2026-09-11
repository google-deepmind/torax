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

"""iterhybrid_predictor_corrector with small fixed dt and time-dependent geo.

Exercises the time interpolation of geometry on every time step, which is
pre-computed when using the fixed time step calculator.
"""

import copy

from torax.benchmarks import iterhybrid_predictor_corrector_fixed_small_dt

CONFIG = copy.deepcopy(iterhybrid_predictor_corrector_fixed_small_dt.CONFIG)
CONFIG['numerics']['adaptive_dt'] = False
CONFIG['geometry'] = {
    'geometry_type': 'chease',
    'Ip_from_parameters': True,
    'geometry_configs': {
        0.0: {
            'geometry_file': 'iterhybrid.mat2cols',
            'R_major': 6.2,
            'a_minor': 2.0,
            'B_0': 5.3,
        },
        5.0: {
            'geometry_file': 'iterhybrid.mat2cols',
            'R_major': 6.3,
            'a_minor': 2.0,
            'B_0': 5.3,
        },
    },
}
