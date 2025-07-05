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

"""Pydantic utilities and base classes."""

import copy
import functools
from typing import Any, Callable, TypeAlias
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
from typing_extensions import Annotated

TIME_INVARIANT = model_base.TIME_INVARIANT
JAX_STATIC = model_base.JAX_STATIC

# Physical units.
# keep-sorted start
CubicMeter: TypeAlias = pydantic.PositiveFloat
GreenwaldFraction: TypeAlias = pydantic.PositiveFloat
KiloElectronVolt: TypeAlias = pydantic.PositiveFloat
Meter: TypeAlias = pydantic.PositiveFloat
MeterPerSecond: TypeAlias = float
MeterSquaredPerSecond: TypeAlias = pydantic.NonNegativeFloat
Pascal: TypeAlias = pydantic.PositiveFloat
PositiveMeterSquaredPerSecond: TypeAlias = pydantic.PositiveFloat
# Time can sometimes be 0, eg. for the start of an interval.
Second: TypeAlias = pydantic.NonNegativeFloat
Tesla: TypeAlias = pydantic.PositiveFloat
# keep-sorted end
Density: TypeAlias = CubicMeter | GreenwaldFraction

UnitInterval: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]
OpenUnitInterval: TypeAlias = Annotated[float, pydantic.Field(gt=0.0, lt=1.0)]

NumpyArray = pydantic_types.NumpyArray
NumpyArray1D = pydantic_types.NumpyArray1D
NumpyArray1DSorted = pydantic_types.NumpyArray1DSorted

BaseModelFrozen = model_base.BaseModelFrozen

TimeVaryingScalar = interpolated_param_1d.TimeVaryingScalar
TimeVaryingArray = interpolated_param_2d.TimeVaryingArray
PositiveTimeVaryingScalar = interpolated_param_1d.PositiveTimeVaryingScalar
UnitIntervalTimeVaryingScalar = (
    interpolated_param_1d.UnitIntervalTimeVaryingScalar
)
PositiveTimeVaryingArray = interpolated_param_2d.PositiveTimeVaryingArray

ValidatedDefault = functools.partial(pydantic.Field, validate_default=True)

Grid1D = interpolated_param_2d.Grid1D
set_grid = interpolated_param_2d.set_grid


def create_default_model_injector(
    discriminator_name: str, model_name: str
) -> Callable[[Any], Any]:
  """A factory that creates a Pydantic `BeforeValidator` function.

  The returned validator is designed for discriminated unions. It checks if the
  input is a dictionary lacking a discriminator_name key. If so, it injects the
  provided `model_name` into a copy of the dictionary as a default
  discriminator_name,

  Args:
    discriminator_name: The discriminator string.
    model_name: The default string to inject into the discriminator_name key.

  Returns:
    A validator function suitable for use with `pydantic.BeforeValidator`.
  """

  def validator(v: Any) -> Any:
    if isinstance(v, dict) and discriminator_name not in v:
      # Work on a copy to avoid mutating the original input dict
      v_copy = copy.deepcopy(v)
      v_copy[discriminator_name] = model_name
      return v_copy
    return v

  return validator
