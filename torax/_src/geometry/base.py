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
"""Base class for geometry configuration."""

from typing import Annotated

import pydantic
from torax._src.torax_pydantic import torax_pydantic


class BaseGeometryConfig(torax_pydantic.BaseModelFrozen):
  """Base class for all geometry configuration classes.

  Attributes:
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
  """

  n_rho: Annotated[pydantic.PositiveInt, torax_pydantic.TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
