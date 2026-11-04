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

"""GeometryModel and PrescribedGeometryModel abstractions for TORAX."""

# pylint: disable=invalid-name

import abc
import dataclasses

import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry
from torax._src.geometry import geometry_state as geometry_state_lib
from torax._src.geometry import runtime_params as geometry_runtime_params
import xarray as xr


@dataclasses.dataclass(frozen=True, eq=False)
class GeometryModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for geometry models."""

  @abc.abstractmethod
  def initialize(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      *,
      core_profiles: core_profiles_lib.CoreProfiles | None = None,
  ) -> tuple[geometry.Geometry, geometry_state_lib.GeometryState]:
    """Evaluates initial `Geometry` and `GeometryState` at `t_initial`."""

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> geometry_state_lib.GeometryState:
    """Reconstructs GeometryState from a `/model_state/geometry` DataTree."""
    del datatree
    _, state = self.initialize(runtime_params)
    return state

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry_state: geometry_state_lib.GeometryState,
      *,
      dt: array_typing.FloatScalar,
      core_profiles: core_profiles_lib.CoreProfiles | None = None,
  ) -> tuple[geometry.Geometry, geometry_state_lib.GeometryState]:
    """Evaluates `Geometry` and updated `GeometryState` for a step of size `dt`."""


@dataclasses.dataclass(frozen=True, eq=False)
class PrescribedGeometryModel(GeometryModel):
  """Evaluates prescribed Geometry from PrescribedGeometryRuntimeParams."""

  def initialize(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      *,
      core_profiles: core_profiles_lib.CoreProfiles | None = None,
  ) -> tuple[geometry.Geometry, geometry_state_lib.GeometryState]:
    """Evaluates initial Geometry and GeometryState at t_initial."""
    del core_profiles
    if not isinstance(
        runtime_params.geometry,
        geometry_runtime_params.PrescribedGeometryRuntimeParams,
    ):
      raise ValueError(
          'Expected PrescribedGeometryRuntimeParams in'
          f' runtime_params.geometry, got {type(runtime_params.geometry)}.'
      )
    geo = runtime_params.geometry.geometry
    phi_b_dot = jnp.where(
        runtime_params.geometry.calc_Phi_b_dot,
        runtime_params.geometry.initial_Phi_b_dot,
        jnp.zeros((), dtype=jax_utils.get_dtype()),
    )
    init_state = geometry_state_lib.GeometryState.from_geometry(
        geo, Phi_b_dot=phi_b_dot
    )
    return geo, init_state

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry_state: geometry_state_lib.GeometryState,
      *,
      dt: array_typing.FloatScalar,
      core_profiles: core_profiles_lib.CoreProfiles | None = None,
  ) -> tuple[geometry.Geometry, geometry_state_lib.GeometryState]:
    """Evaluates Geometry and updated GeometryState for a step of size dt."""
    del core_profiles
    if not isinstance(
        runtime_params.geometry,
        geometry_runtime_params.PrescribedGeometryRuntimeParams,
    ):
      raise ValueError(
          'Expected PrescribedGeometryRuntimeParams in'
          f' runtime_params.geometry, got {type(runtime_params.geometry)}.'
      )
    geo = runtime_params.geometry.geometry
    updated_state = geometry_state_lib.update_geometry_state(
        geometry_state=geometry_state,
        geo=geo,
        dt=dt,
        calc_Phi_b_dot=runtime_params.geometry.calc_Phi_b_dot,
    )
    return geo, updated_state
