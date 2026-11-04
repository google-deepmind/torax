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
"""Base class for neoclassical poloidal velocity models."""

from __future__ import annotations

import abc
import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src.core_profiles import solver_profiles as solver_profiles_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class PoloidalVelocity:
  """Values returned by a poloidal velocity model."""

  poloidal_velocity: cell_variable.CellVariable

  @classmethod
  def from_arrays(
      cls,
      geometry: geometry_lib.Geometry,
      poloidal_velocity: array_typing.FloatVectorCell,
      right_face_constraint: array_typing.FloatScalar | None = None,
  ) -> Self:
    """Constructs a PoloidalVelocity object from arrays.

    Args:
      geometry: The magnetic geometry of the tokamak.
      poloidal_velocity: Poloidal velocity profile on the cell grid [m/s].
      right_face_constraint: Optional right boundary condition for poloidal
        velocity [m/s]. Defaults to 0.0.

    Returns:
      A `PoloidalVelocity` instance with a `CellVariable` profile for
      `poloidal_velocity`.
    """
    poloidal_velocity_cell_var = cell_variable.CellVariable(
        value=poloidal_velocity,
        face_centers=geometry.rho_face_norm,
        right_face_constraint=(
            right_face_constraint
            if right_face_constraint is not None
            else jnp.zeros(())
        ),
        right_face_grad_constraint=None,
    )
    return cls(poloidal_velocity=poloidal_velocity_cell_var)

  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> Self:
    """Returns a PoloidalVelocity with all values set to zero."""
    return cls.from_arrays(
        geometry=geometry,
        poloidal_velocity=jnp.zeros_like(geometry.rho_norm),
    )


class PoloidalVelocityModel(abc.ABC):
  """Base class for poloidal velocity models."""

  @abc.abstractmethod
  def calculate_poloidal_velocity(
      self,
      runtime_params: poloidal_velocity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      solver_profiles: solver_profiles_lib.SolverProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> PoloidalVelocity:
    """Calculates neoclassical poloidal velocity."""


class PoloidalVelocityModelConfig(torax_pydantic.ConfigBase):
  """Base class for poloidal velocity model configs.

  Attributes:
    poloidal_velocity_multiplier: Multiplier applied to the neoclassical
      poloidal velocity.
  """

  poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0

  def build_runtime_params(
      self,
  ) -> poloidal_velocity_runtime_params.RuntimeParams:
    """Builds runtime params."""
    return poloidal_velocity_runtime_params.RuntimeParams(
        poloidal_velocity_multiplier=self.poloidal_velocity_multiplier,
    )

  @abc.abstractmethod
  def build_model(self) -> PoloidalVelocityModel:
    """Builds poloidal velocity model."""
