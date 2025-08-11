# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common types for using jaxtyping in TORAX."""
from typing import TypeAlias
import jax
import jaxtyping as jt
import numpy as np
import typeguard

Array: TypeAlias = jax.Array | np.ndarray

ScalarFloat: TypeAlias = jt.Float[Array | float, ""]
ScalarBool: TypeAlias = jt.Bool[Array | bool, ""]
ScalarInt: TypeAlias = jt.Int[Array | int, ""]

ArrayFloat: TypeAlias = jt.Float[Array, "rhon"]
ArrayBool: TypeAlias = jt.Bool[Array, "rhon"]

jaxtyped = jt.jaxtyped(typechecker=typeguard.typechecked)
