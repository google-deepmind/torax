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

"""Base class for bootstrap current models."""
import abc

import chex
import jax
import jax.numpy as jnp
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import runtime_params
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@chex.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
  """Values returned by a bootstrap current model."""
  j_bootstrap: jax.Array
  j_bootstrap_face: jax.Array

  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> 'BootstrapCurrent':
    """Returns a BootstrapCurrent with all values set to zero."""
    return cls(
        j_bootstrap=jnp.zeros_like(geometry.rho_norm),
        j_bootstrap_face=jnp.zeros_like(geometry.rho_face_norm),
    )


class BootstrapCurrentModel(abc.ABC):
  """Base class for bootstrap current models."""

  @abc.abstractmethod
  def calculate_bootstrap_current(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> BootstrapCurrent:
    """Calculates bootstrap current."""


class BootstrapCurrentModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for bootstrap current model configs."""

  @abc.abstractmethod
  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    """Builds dynamic runtime params."""

  @abc.abstractmethod
  def build_model(self) -> BootstrapCurrentModel:
    """Builds bootstrap current model."""
