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

"""Common functions and classes for interpolated parameters."""

import abc
import functools
import chex
import pydantic
from torax import interpolated_param
from torax.torax_pydantic import model_base
from typing_extensions import Self


class TimeVaryingBase(model_base.BaseModelMutable):
  """Base class for time varying interpolated parameters."""

  def get_value(self, x: chex.Numeric) -> chex.Array:
    """Returns the value of this parameter interpolated at x=time.

    Requires self.grid to be set.

    Args:
      x: An array of times to interpolate at.

    Returns:
      An array of interpolated values.
    """
    return self._get_cached_interpolated_param.get_value(x)

  def __eq__(self, other):
    """Custom equality check."""

    try:
      chex.assert_trees_all_equal(vars(self), vars(other))
      return True
    except AssertionError:
      return False

  @functools.cached_property
  @abc.abstractmethod
  def _get_cached_interpolated_param(
      self,
  ) -> (
      interpolated_param.InterpolatedVarSingleAxis
      | interpolated_param.InterpolatedVarTimeRho
  ):
    """Returns the value of this parameter interpolated at x=time."""
    ...

  @pydantic.model_validator(mode='after')
  def clear_cached_property(self) -> Self:
    try:
      del self._get_cached_interpolated_param
    except AttributeError:
      pass
    return self
