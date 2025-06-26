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

"""Base class for neoclassical transport models."""
import abc

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import runtime_params
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax_utils.jax_dataclass(kw_only=True, frozen=True)
class NeoclassicalTransport:
  """Values returned by a neoclassical transport model.

  Attributes:
    chi_neo_i: Ion neoclassical heat transport coefficient [m^2/s].
    chi_neo_e: Electron neoclassical heat transport coefficient [m^2/s].
    D_neo_e: Electron neoclassical particle transport coefficient [m^2/s].
    V_neo_e: Electron neoclassical convection velocity [m/s]. Includes all terms
      apart from the Ware Pinch.
    V_neo_ware_e: Electron Ware Pinch velocity [m/s]. This is the component of
      the neoclassical convection that is dependent on the parallel electric
      field. It is separated from V_neo_e for interpretation convenience.
  """
  chi_neo_i: jax.Array
  chi_neo_e: jax.Array
  D_neo_e: jax.Array
  V_neo_e: jax.Array
  V_neo_ware_e: jax.Array

  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> 'NeoclassicalTransport':
    """Returns a NeoclassicalTransport with all values set to zero."""
    return cls(
        chi_neo_i=jnp.zeros_like(geometry.rho_face_norm),
        chi_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        D_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        V_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        V_neo_ware_e=jnp.zeros_like(geometry.rho_face_norm),
    )


class NeoclassicalTransportModel(abc.ABC):
  """Base class for neoclassical transport models."""

  @abc.abstractmethod
  def calculate_neoclassical_transport(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> NeoclassicalTransport:
    """Calculates neoclassical transport."""


class NeoclassicalTransportModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for neoclassical transport model configs."""

  @abc.abstractmethod
  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    """Builds dynamic runtime params."""

  @abc.abstractmethod
  def build_model(self) -> NeoclassicalTransportModel:
    """Builds neoclassical transport model."""
