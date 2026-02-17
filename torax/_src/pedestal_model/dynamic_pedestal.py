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
"""A pedestal that forms dynamically based on the LH threshold and critical ballooning parameter."""

import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.physics import formulas
from torax._src.physics import scaling_laws

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params_lib.RuntimeParams):
  """Runtime params for the DynamicPedestalModel."""

  suppression_factor: array_typing.FloatScalar
  suppression_rate: array_typing.FloatScalar
  augmentation_factor: array_typing.FloatScalar
  augmentation_rate: array_typing.FloatScalar
  alpha_crit: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class DynamicPedestal(pedestal_model.PedestalModel):
  """A pedestal that forms dynamically based on the LH threshold and critical ballooning parameter."""

  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    if (
        runtime_params.pedestal.mode
        != pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    ):
      raise ValueError('DynamicPedestal only supports ADAPTIVE_TRANSPORT mode.')

    pedestal_runtime_params = runtime_params.pedestal
    assert isinstance(pedestal_runtime_params, RuntimeParams)

    # Get the pedestal top location.
    rho_norm_ped_top_idx = jnp.abs(
        geo.rho_norm - pedestal_runtime_params.rho_norm_ped_top
    ).argmin()
    rho_norm_ped_top = jax.lax.dynamic_index_in_dim(
        geo.rho_norm,
        rho_norm_ped_top_idx,
        keepdims=False,
    )
    pedestal_active_mask_face = jnp.where(
        geo.rho_face_norm >= rho_norm_ped_top, 1.0, 0.0
    )

    # Are we above P_LH? If so, decrease chi
    _, _, P_LH, _ = scaling_laws.calculate_plh_scaling_factor(
        geo, core_profiles
    )
    # TODO(b/323504363): use the correct source profiles to calculate P_SOL_total.
    # dP_e_drho_norm = merged_source_profiles.total_sources('T_e', geo)
    # dP_i_drho_norm = merged_source_profiles.total_sources('T_i', geo)
    # Integrate over rho_norm to get total power out of the separatrix [W].
    # P_SOL_total = math_utils.volume_integration(
    #     dP_e_drho_norm + dP_i_drho_norm, geo
    # )
    P_SOL_total = P_LH  # TODO(b/323504363): replace with calculated value
    # We use a sigmoid function to smooth the transition.
    # If P < P_LH, h_mode_weight -> 0, transport_decrease_multiplier -> 1.0.
    # If P > P_LH, h_mode_weight -> 1, transport_decrease_multiplier ->
    # suppression_factor.
    h_mode_weight = jax.nn.sigmoid(
        (P_SOL_total - P_LH) / (pedestal_runtime_params.suppression_rate * P_LH)
    )
    transport_decrease_multiplier = (
        1.0 - h_mode_weight
    ) * 1.0 + h_mode_weight * pedestal_runtime_params.suppression_factor

    # Are we above the critical ballooning parameter anywhere in the
    # pedestal region? If so, increase chi.
    dp_dr_face = formulas.calc_pprime(core_profiles)
    alpha_face = jnp.abs(
        2
        * constants.CONSTANTS.mu_0
        * geo.R_major_profile_face
        * core_profiles.q_face**2
        / geo.B_0**2
        * dp_dr_face
    )
    max_alpha = jnp.max(pedestal_active_mask_face * alpha_face)
    # We use a softplus function to smooth the transition.
    # If max_alpha < alpha_crit, continuous_elm_weight -> 0,
    # transport_increase_multiplier -> 1.0
    # If max_alpha > alpha_crit, continuous_elm_weight -> inf,
    # transport_increase_multiplier -> inf.
    continuous_elm_weight = jax.nn.softplus(
        (max_alpha - pedestal_runtime_params.alpha_crit)
        / (
            pedestal_runtime_params.augmentation_rate
            * pedestal_runtime_params.alpha_crit
        )
    )
    transport_increase_multiplier = 1.0 + (
        continuous_elm_weight * pedestal_runtime_params.augmentation_factor
    )

    # Combine the multipliers.
    transport_multiplier = jnp.exp(
        jnp.log(transport_decrease_multiplier)
        + jnp.log(transport_increase_multiplier)
    )

    # For simplicity, we currently scale all coefficients by the same factor.
    return pedestal_model.AdaptiveTransportPedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        chi_e_multiplier=transport_multiplier,
        chi_i_multiplier=transport_multiplier,
        D_e_multiplier=transport_multiplier,
        v_e_multiplier=transport_multiplier,
    )
