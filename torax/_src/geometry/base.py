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
from typing import Annotated, Any

import numpy as np
import pydantic
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions


class BaseGeometryConfig(torax_pydantic.BaseModelFrozen):
  """Base class for all geometry configuration classes.

  Attributes:
    n_rho: Number of radial grid cells in a uniform grid. Must be at least 4.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
      For a grid with N cells, there are N+1 faces. This can be non-uniform.
      The internal TORAX fvm method assumes we have at least 4 cells so this
      will be validated here.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
  """

  n_rho: Annotated[int | None, torax_pydantic.TIME_INVARIANT] = None
  face_centers: Annotated[
      torax_pydantic.NumpyArray1DSorted | None, torax_pydantic.TIME_INVARIANT
  ] = None
  hires_factor: pydantic.PositiveInt = 4

  @pydantic.model_validator(mode='before')
  @classmethod
  def _validate_inputs(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data
    # Set default n_rho if not provided.
    if 'n_rho' not in data and 'face_centers' not in data:
      data['n_rho'] = 25
    return data

  @pydantic.model_validator(mode='after')
  def _validate_n_rho_or_face_centers(self) -> typing_extensions.Self:
    """Validates that there are at least 4 cells."""
    if self.n_rho is None and self.face_centers is None:
      raise ValueError('Either n_rho or face_centers must be set.')

    if self.face_centers is not None:
      if len(self.face_centers) < 5:
        raise ValueError(
            'face_centers must have at least 5 elements (4 cells). Got'
            f' {len(self.face_centers)}'
        )
      if not np.isclose(self.face_centers[0], 0.0) or not np.isclose(
          self.face_centers[-1], 1.0
      ):
        raise ValueError(
            'face_centers must start at 0.0 and end at 1.0. Got'
            f' {self.face_centers}'
        )

    if self.n_rho is not None and self.n_rho < 4:
      raise ValueError('n_rho must be at least 4')

    return self

  def get_face_centers(self) -> np.ndarray:
    """Returns face_centers, computing from n_rho if needed."""
    if self.face_centers is not None:
      return self.face_centers
    return interpolated_param_2d.get_face_centers(self.n_rho)

  def __eq__(self, other: typing_extensions.Self) -> bool:
    """Equality operator for BaseGeometryConfig."""
    if not isinstance(other, type(self)):
      return False
    return (
        np.array_equal(self.get_face_centers(), other.get_face_centers())
        and self.hires_factor == other.hires_factor
    )
