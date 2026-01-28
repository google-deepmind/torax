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

"""TGLFNNukaea transport with rotation enabled."""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector_tglfnn_ukaea

# Note: This simulation uses a D-T plasma, but TGLFNNukaea is currently
# only designed for D only
CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector_tglfnn_ukaea.CONFIG)

CONFIG['transport']['use_rotation'] = True

# Reference angular velocity used for mach number calculations.
# omega_ref ~ sqrt(2 * T_i/ m_i) / R_major
# We use T_i ~20 keV, R_major=6.2 and m_i = m_DT which gives v_ref ~1.5e5
_omega_ref = 1.5e5
CONFIG['profile_conditions']['toroidal_angular_velocity'] = {
    0.0: 1.8 * _omega_ref,
    1.0: 0.0,
}
