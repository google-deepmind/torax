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
"""Custom pedestal model allowing user-defined callable functions.

This module provides a flexible API for users to define custom pedestal
scaling laws without modifying the TORAX source code. Users can provide
callable functions that compute pedestal values based on runtime parameters,
geometry, and core profiles.
"""
import dataclasses
from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from typing_extensions import override


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime params for the CustomPedestalModel.

  Attributes:
    rho_norm_ped_top: The location of the pedestal top. Can be a scalar value
      or will be computed by rho_norm_ped_top_fn if provided.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW (Greenwald fraction).
  """

  rho_norm_ped_top: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


@dataclasses.dataclass(frozen=True, eq=False)
class CustomPedestalModel(pedestal_model.PedestalModel):
  """Custom pedestal model using user-defined callable functions.

  This model allows users to specify custom functions for computing pedestal
  temperature and density values. This is useful for implementing
  machine-specific scaling laws (e.g., STEP pedestal models with Europed data
  fits) without modifying the TORAX source code.

  The callable functions receive runtime parameters, geometry, and core
  profiles as inputs and should return scalar values for the respective
  pedestal quantities.

  Attributes:
    T_i_ped_fn: Callable function to compute ion temperature at pedestal [keV].
      Signature: (runtime_params, geo, core_profiles) -> FloatScalar
    T_e_ped_fn: Callable function to compute electron temperature at pedestal
      [keV]. Signature: (runtime_params, geo, core_profiles) -> FloatScalar
    n_e_ped_fn: Callable function to compute electron density at pedestal
      [m^-3 or fGW]. Signature: (runtime_params, geo, core_profiles) ->
      FloatScalar
    rho_norm_ped_top_fn: Optional callable function to compute pedestal top
      location. If None, uses the value from RuntimeParams. Signature:
      (runtime_params, geo, core_profiles) -> FloatScalar

  Example:
    ```python
    # Define custom scaling functions
    def custom_T_e_ped(runtime_params, geo, core_profiles):
      # Example: EPED-like scaling
      Ip_MA = runtime_params.profile_conditions.Ip / 1e6
      B_T = geo.B0
      return 0.5 * (Ip_MA ** 0.2) * (B_T ** 0.8)

    def custom_n_e_ped(runtime_params, geo, core_profiles):
      # Example: Return as Greenwald fraction
      return 0.7  # 0.7 * nGW

    def custom_T_i_ped(runtime_params, geo, core_profiles):
      T_e = custom_T_e_ped(runtime_params, geo, core_profiles)
      return 1.2 * T_e  # T_i = 1.2 * T_e

    # Create the model
    model = CustomPedestalModel(
        T_i_ped_fn=custom_T_i_ped,
        T_e_ped_fn=custom_T_e_ped,
        n_e_ped_fn=custom_n_e_ped,
    )
    ```
  """

  T_i_ped_fn: Callable[
      [runtime_params_slice.RuntimeParams, geometry.Geometry, state.CoreProfiles],
      array_typing.FloatScalar,
  ]
  T_e_ped_fn: Callable[
      [runtime_params_slice.RuntimeParams, geometry.Geometry, state.CoreProfiles],
      array_typing.FloatScalar,
  ]
  n_e_ped_fn: Callable[
      [runtime_params_slice.RuntimeParams, geometry.Geometry, state.CoreProfiles],
      array_typing.FloatScalar,
  ]
  rho_norm_ped_top_fn: Callable[
      [runtime_params_slice.RuntimeParams, geometry.Geometry, state.CoreProfiles],
      array_typing.FloatScalar,
  ] | None = None

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    """Compute pedestal values using user-defined functions."""
    pedestal_params = runtime_params.pedestal
    assert isinstance(pedestal_params, RuntimeParams)

    # Compute pedestal location
    if self.rho_norm_ped_top_fn is not None:
      rho_norm_ped_top = self.rho_norm_ped_top_fn(
          runtime_params, geo, core_profiles
      )
    else:
      rho_norm_ped_top = pedestal_params.rho_norm_ped_top

    # Compute pedestal index
    rho_norm_ped_top_idx = jnp.abs(geo.rho_norm - rho_norm_ped_top).argmin()

    # Compute temperatures using user-provided functions
    T_i_ped = self.T_i_ped_fn(runtime_params, geo, core_profiles)
    T_e_ped = self.T_e_ped_fn(runtime_params, geo, core_profiles)

    # Compute density using user-provided function
    n_e_ped_raw = self.n_e_ped_fn(runtime_params, geo, core_profiles)

    # Convert n_e_ped to absolute units if provided as Greenwald fraction
    nGW = (
        runtime_params.profile_conditions.Ip
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_ped = jnp.where(
        pedestal_params.n_e_ped_is_fGW,
        n_e_ped_raw * nGW,
        n_e_ped_raw,
    )

    return pedestal_model.PedestalModelOutput(
        n_e_ped=n_e_ped,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
    )
