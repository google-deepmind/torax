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
"""GACODE NEO surrogate bootstrap current model for TORAX."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal, Optional

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_current_runtime_params
from torax._src.neoclassical.formulas import neo_surrogate as neo_formulas
from torax._src.neoclassical.transport import neo_surrogate as neo_transport
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(bootstrap_current_runtime_params.RuntimeParams):
  """Runtime parameters for the NEO surrogate bootstrap current model."""

  axis_extrapolate: array_typing.BoolScalar = True


class NeoSurrogateBootstrapCurrentModelConfig(
    bootstrap_current_base.BootstrapCurrentModelConfig
):
  """Pydantic model config for NEO surrogate bootstrap current."""

  model_name: Annotated[
      Literal['neo_surrogate'], torax_pydantic.JAX_STATIC
  ] = 'neo_surrogate'
  axis_extrapolate: bool = True

  @override
  def build_model(self) -> NeoSurrogateBootstrapCurrentModel:
    return NeoSurrogateBootstrapCurrentModel()

  @override
  def build_runtime_params(self) -> RuntimeParams:
    return RuntimeParams(
        bootstrap_multiplier=self.bootstrap_multiplier,
        axis_extrapolate=self.axis_extrapolate,
    )


class NeoSurrogateBootstrapCurrentModel(
    bootstrap_current_base.BootstrapCurrentModel
):
  """GACODE NEO surrogate bootstrap current model."""

  def __init__(
      self,
      base_params: Optional[neo_formulas.BaseTransportParams] = None,
  ):
    if base_params is None:
      base_params = neo_formulas.get_default_base_transport_params()
    self.base_params = base_params

  @override
  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> bootstrap_current_base.BootstrapCurrent:
    """Calculates parallel bootstrap current density on cell and face grids."""
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(bootstrap_params, RuntimeParams)

    # Use default transport extraction settings for input features
    dummy_transport_params = neo_transport.RuntimeParams(
        axis_extrapolate=bootstrap_params.axis_extrapolate,
        min_epsilon=1e-4,
        chi_min=0.0,
        chi_max=100.0,
        D_e_min=0.0,
        D_e_max=100.0,
        V_e_min=-50.0,
        V_e_max=50.0,
        chi_neo_i_multiplier=1.0,
        chi_neo_e_multiplier=1.0,
        D_neo_e_multiplier=1.0,
        V_neo_e_multiplier=1.0,
        V_neo_ware_e_multiplier=1.0,
    )
    x_base = neo_transport.extract_base_features(
        dummy_transport_params, geometry, core_profiles
    )
    out = neo_formulas.predict_base_transport(self.base_params, x_base)
    j_bs_surr = out[..., 5]

    j_bs_extrap = j_bs_surr.at[0].set(j_bs_surr[1])
    j_bs_surr = jnp.where(
        bootstrap_params.axis_extrapolate,
        j_bs_extrap,
        j_bs_surr,
    )

    # Convert to physical current density [A/m^2] matching Sauter/TORAX
    # geometric projection: prefactor = -F_face * 2pi / B_0 / (dpsi/drnorm).
    dpsi_drnorm = core_profiles.psi.face_grad()
    prefactor = -geometry.F_face * 2.0 * jnp.pi / geometry.B_0
    global_coeff = jnp.where(
        jnp.abs(dpsi_drnorm) > 1e-7,
        jnp.abs(prefactor / dpsi_drnorm),
        0.0,
    )
    Te_face_keV = jnp.maximum(core_profiles.T_e.face_value(), 1e-3)
    ne_face = jnp.maximum(core_profiles.n_e.face_value(), 1e16)
    pe_face = Te_face_keV * 1e3 * neo_formulas.Q_E * ne_face
    mult = bootstrap_params.bootstrap_multiplier

    j_bs_face = j_bs_surr * mult * pe_face * global_coeff
    j_bs_cell = geometry_lib.face_to_cell(j_bs_face)

    return bootstrap_current_base.BootstrapCurrent(
        j_parallel_bootstrap=j_bs_cell,
        j_parallel_bootstrap_face=j_bs_face,
    )

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)
