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

import abc
from typing import Annotated, Any, Self

import numpy as np
import pydantic
from torax._src.geometry import trapped_fraction
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic

TrappedFractionSource = trapped_fraction.TrappedFractionSource


class GeometryConfigBase(torax_pydantic.ConfigBase):
  """Base class for all geometry configuration classes.

  Attributes:
    n_rho: Number of radial grid cells. Creates a uniform grid with cell width
      $drho_norm = 1 / n_rho$. Must be at least 4. Defaults to `25` when neither
      `n_rho` nor `face_centers` is specified. Either `n_rho` or `face_centers`
      must be specified.
    face_centers: Explicit array of face center coordinates [$rho_norm$]
      (ranging from 0 to 1) for defining non-uniform radial grids.
      For a grid with $N$ cells, there must be $N + 1$ face centers. The array
      must start at `0.0`, end at `1.0`, be strictly increasing, and have at
      least 5 elements (4 cells). This enables finer resolution in regions of
      interest, such as near the plasma edge:

      ```
      import numpy as np

      # Coarse in core (8 cells in [0, 0.8]),
      # fine near edge (8 cells in [0.8, 1]).
      core_faces = np.linspace(0, 0.8, 9)
      edge_faces = np.linspace(0.8, 1.0, 9)[1:]
      face_centers = np.concatenate([core_faces, edge_faces])

      'geometry': {
          'geometry_type': 'chease',
          'face_centers': face_centers,
      }
      ```
    hires_factor: Only used when the initial condition `psi` is from plasma
      current. Sets up a higher resolution mesh with `nrho_hires = n_rho *
      hires_factor`, used for `j` to `psi` conversions.
    trapped_fraction_source: Selects how the effective trapped particle
      fraction, used by the neoclassical `bootstrap_current`, `conductivity`,
      and `transport` models, is computed. Computed once at geometry
      construction time. Options are:

      - `'sauter'` (default): Uses the analytical approximation from
        |sauter16|. Supported by every geometry source.
      - `'file'`: Reads the value precomputed by the input equilibrium code
        directly from the geometry file. Only supported for CHEASE and IMAS
        geometries.
      - `'exact'`: Computes the exact bounce-averaged integral directly from
        the traced 2D equilibrium, using the formulation of
        |lin-liu-miller1995|. Only supported for EQDSK and IMAS geometries.
  """  # fmt: skip

  @property
  @abc.abstractmethod
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction.TrappedFractionSource]:
    """Supported trapped fraction sources for this geometry."""

  n_rho: Annotated[int | None, torax_pydantic.TIME_INVARIANT] = None
  face_centers: Annotated[
      torax_pydantic.NumpyArray1DSorted | None, torax_pydantic.TIME_INVARIANT
  ] = None
  hires_factor: pydantic.PositiveInt = 4
  trapped_fraction_source: Annotated[
      trapped_fraction.TrappedFractionSource, torax_pydantic.TIME_INVARIANT
  ] = trapped_fraction.TrappedFractionSource.SAUTER

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
  def _validate_n_rho_or_face_centers(self) -> Self:
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

  @pydantic.model_validator(mode='after')
  def _validate_trapped_fraction_source(self) -> Self:
    """Validates that trapped_fraction_source is supported by this geometry."""
    if (
        self.trapped_fraction_source
        not in self._supported_trapped_fraction_sources
    ):
      allowed = ', '.join(
          sorted(s.value for s in self._supported_trapped_fraction_sources)
      )
      raise ValueError(
          f'trapped_fraction_source={self.trapped_fraction_source.value} is'
          f' not supported for {type(self).__name__}. Supported options:'
          f' {allowed}.'
      )
    return self

  def get_face_centers(self) -> np.ndarray:
    """Returns face_centers, computing from n_rho if needed."""
    if self.face_centers is not None:
      return self.face_centers
    return interpolated_param_2d.get_face_centers(self.n_rho)

  def get_grid(self) -> torax_pydantic.Grid1D:
    """Returns the radial `Grid1D` for this geometry configuration."""
    return torax_pydantic.Grid1D(face_centers=self.get_face_centers())

  def __eq__(self, other: object) -> bool:
    """Equality operator for GeometryConfigBase."""
    if not isinstance(other, GeometryConfigBase) or type(other) is not type(
        self
    ):
      return False
    return (
        np.array_equal(self.get_face_centers(), other.get_face_centers())
        and self.hires_factor == other.hires_factor
    )
