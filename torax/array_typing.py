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
"""Common types and helpers for using jaxtyping in TORAX."""

from typing import Callable, TypeVar
import chex
import jaxtyping as jt
import typeguard

F = TypeVar("F", bound=Callable)
ScalarFloat = jt.Float[chex.Array | float, ""]
ScalarBool = jt.Bool[chex.Array | bool, ""]
ArrayFloat = jt.Float[chex.Array, "rhon"]
ArrayBool = jt.Bool[chex.Array, "rhon"]


def typed(function: F) -> F:
  """Helper decorator for using jaxtyping for shape and dtype checking.

  Example usage:
  @typed
  def f(x: ScalarFloat) -> ScalarFloat:
    ...

  Jaxtyping is enabled by default, to globally disable jaxtyping set
  JAXTYPING_DISABLE=True.

  Args:
    function: The function to shape check.

  Returns:
    The decorated function.
  """
  return jt.jaxtyped(function, typechecker=typeguard.typechecked)
