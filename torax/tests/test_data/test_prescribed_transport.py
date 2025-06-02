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

"""Tests prescribed transport.

Chi:
- Exponential on-axis, mimicking neoclassical contribution
- Gaussian off-axis, mimicking Bohm-like transport. Moves from rho_norm=0.7 to
  rho_norm=0.5 over the course of the simulation (0-5s).

De, Ve:
- Constant (defaults)
"""

import copy
from torax.tests.test_data import default_config
import jax.numpy as jnp

x = jnp.linspace(0, 1, 10)
chi_base_t1 = jnp.exp(-20 * x) + 0.3 * jnp.exp(-(((x - 0.7) / 0.3) ** 2))
chi_base_t2 = jnp.exp(-20 * x) + 0.3 * jnp.exp(-(((x - 0.5) / 0.3) ** 2))

CONFIG = copy.deepcopy(default_config.CONFIG)

CONFIG['transport'] = {
    # Note: prescribed transport is currently handled by ConstantTransportModel
    # The name of this model will change in a future release
    'model_name': 'constant',
    'chi_i': (
        jnp.array([0.0, 5.0]),
        x,
        1.5 * jnp.stack([chi_base_t1, chi_base_t2]),
    ),
    'chi_e': (
        jnp.array([0.0, 5.0]),
        x,
        jnp.stack([chi_base_t1, chi_base_t2]),
    ),
    'D_e': 1.0,
    'V_e': -0.33,
}
