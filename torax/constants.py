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

"""Physics constants.

This module saves immutable constants used in various calculations.
"""

from typing import Final

import chex
from jax import numpy as jnp


@chex.dataclass(frozen=True)
class Constants:
  keV2J: chex.Numeric  # pylint: disable=invalid-name
  mp: chex.Numeric
  qe: chex.Numeric
  me: chex.Numeric
  epsilon0: chex.Numeric
  mu0: chex.Numeric
  eps: chex.Numeric


CONSTANTS: Final[Constants] = Constants(
    keV2J=1e3 * 1.6e-19,
    mp=1.67e-27,
    qe=1.6e-19,
    me=9.11e-31,
    epsilon0=8.854e-12,
    mu0=4 * jnp.pi * 1e-7,
    eps=1e-7,
)
