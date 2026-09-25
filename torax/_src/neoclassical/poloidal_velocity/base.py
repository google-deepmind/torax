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
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class PoloidalVelocity:
  """Values returned by a poloidal velocity model."""

  v_pol: cell_variable.CellVariable
  v_pol_face: array_typing.FloatVectorFace

  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> 'PoloidalVelocity':
    """Returns a PoloidalVelocity with all values set to zero."""
    v_pol_face = jnp.zeros_like(geometry.rho_face_norm)
    v_pol = cell_variable.CellVariable(
        value=jnp.zeros_like(geometry.rho_norm),
        face_centers=geometry.rho_face_norm,
        right_face_constraint=v_pol_face[-1],
        right_face_grad_constraint=None,
    )
    return cls(v_pol=v_pol, v_pol_face=v_pol_face)


class PoloidalVelocityModel(abc.ABC):
  """Base class for poloidal velocity models."""

  @abc.abstractmethod
  def calculate_poloidal_velocity(
      self,
      runtime_params: poloidal_velocity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> PoloidalVelocity:
    """Calculates neoclassical poloidal velocity."""


class PoloidalVelocityModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for poloidal velocity model configs."""

  poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0

  @abc.abstractmethod
  def build_runtime_params(
      self,
  ) -> poloidal_velocity_runtime_params.RuntimeParams:
    """Builds runtime params."""

  @abc.abstractmethod
  def build_model(self) -> PoloidalVelocityModel:
    """Builds poloidal velocity model."""
