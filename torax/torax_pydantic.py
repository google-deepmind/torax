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
"""Pydantic utilities and base classes for TORAX."""

# pylint: disable=g-importing-member
from torax._src.torax_pydantic.torax_pydantic import array_bounds_validator
from torax._src.torax_pydantic.torax_pydantic import BaseModelFrozen
from torax._src.torax_pydantic.torax_pydantic import COCOSInt
from torax._src.torax_pydantic.torax_pydantic import CubicMeter
from torax._src.torax_pydantic.torax_pydantic import Density
from torax._src.torax_pydantic.torax_pydantic import GreenwaldFraction
from torax._src.torax_pydantic.torax_pydantic import Grid1D
from torax._src.torax_pydantic.torax_pydantic import JAX_STATIC
from torax._src.torax_pydantic.torax_pydantic import KiloElectronVolt
from torax._src.torax_pydantic.torax_pydantic import Meter
from torax._src.torax_pydantic.torax_pydantic import MeterPerSecond
from torax._src.torax_pydantic.torax_pydantic import MeterSquaredPerSecond
from torax._src.torax_pydantic.torax_pydantic import NonNegativeTimeVaryingArray
from torax._src.torax_pydantic.torax_pydantic import NonNegativeTimeVaryingScalar
from torax._src.torax_pydantic.torax_pydantic import NonNegativeTimeVaryingScalarStep
from torax._src.torax_pydantic.torax_pydantic import NumpyArray
from torax._src.torax_pydantic.torax_pydantic import NumpyArray1D
from torax._src.torax_pydantic.torax_pydantic import NumpyArray1DSorted
from torax._src.torax_pydantic.torax_pydantic import OpenUnitInterval
from torax._src.torax_pydantic.torax_pydantic import Pascal
from torax._src.torax_pydantic.torax_pydantic import PositiveTimeVaryingArray
from torax._src.torax_pydantic.torax_pydantic import PositiveTimeVaryingScalar
from torax._src.torax_pydantic.torax_pydantic import scalar_bounds_validator
from torax._src.torax_pydantic.torax_pydantic import Second
from torax._src.torax_pydantic.torax_pydantic import set_grid
from torax._src.torax_pydantic.torax_pydantic import Tesla
from torax._src.torax_pydantic.torax_pydantic import TIME_INVARIANT
from torax._src.torax_pydantic.torax_pydantic import TimeVaryingArray
from torax._src.torax_pydantic.torax_pydantic import TimeVaryingScalar
from torax._src.torax_pydantic.torax_pydantic import TimeVaryingScalarStep
from torax._src.torax_pydantic.torax_pydantic import UnitInterval
from torax._src.torax_pydantic.torax_pydantic import UnitIntervalTimeVaryingScalar
from torax._src.torax_pydantic.torax_pydantic import ValidatedDefault

__all__ = [
    'BaseModelFrozen',
    'COCOSInt',
    'CubicMeter',
    'Density',
    'GreenwaldFraction',
    'Grid1D',
    'JAX_STATIC',
    'KiloElectronVolt',
    'Meter',
    'MeterPerSecond',
    'MeterSquaredPerSecond',
    'NonNegativeTimeVaryingArray',
    'NonNegativeTimeVaryingScalar',
    'NonNegativeTimeVaryingScalarStep',
    'NumpyArray',
    'NumpyArray1D',
    'NumpyArray1DSorted',
    'OpenUnitInterval',
    'Pascal',
    'PositiveTimeVaryingArray',
    'PositiveTimeVaryingScalar',
    'Second',
    'TIME_INVARIANT',
    'Tesla',
    'TimeVaryingArray',
    'TimeVaryingScalar',
    'TimeVaryingScalarStep',
    'UnitInterval',
    'UnitIntervalTimeVaryingScalar',
    'ValidatedDefault',
    'array_bounds_validator',
    'scalar_bounds_validator',
    'set_grid',
]
