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

import functools
from typing import TypeAlias

import pydantic
from typing_extensions import Annotated

from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types

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
