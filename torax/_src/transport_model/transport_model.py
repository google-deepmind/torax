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

"""The TransportModel class.

A container class that combines transport model components for core and pedestal
regions.
"""

import dataclasses
from typing import Mapping
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.transport_model import component
from torax._src.transport_model import enums
from torax._src.transport_model import (
    runtime_params as transport_runtime_params_lib,
)
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib

MIN_SMOOTHING_WIDTH = 1e-5
# Smoothing kernel values below this fraction of the peak are set to zero,
# limiting the range of the convolution independently of grid resolution.
SMOOTHING_KERNEL_LOWER_CUTOFF = 0.01


@dataclasses.dataclass(frozen=True, eq=False)
class TransportModel(static_dataclass.StaticDataclass):
  """Combines coefficients from a dict of named transport models."""

  core_transport_models: Mapping[str, component.ComponentTransportModel]
  pedestal_transport_models: Mapping[str, component.ComponentTransportModel]

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> transport_state_lib.TransportState:
    """Returns the initial TransportState."""
    return transport_state_lib.TransportState(
        core_transport_states={
            name: model.initial_state(runtime_params)
            for name, model in self.core_transport_models.items()
        },
        pedestal_transport_states={
            name: model.initial_state(runtime_params)
            for name, model in self.pedestal_transport_models.items()
        },
    )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
      two_point_mask: array_typing.BoolVectorFace,
      transport_state: transport_state_lib.TransportState,
  ) -> tuple[
      transport_coeffs.TurbulentTransport, transport_state_lib.TransportState
  ]:
    r"""Calculates transport coefficients and state using TransportModel.

    Combines coefficients from core and pedestal transport models, applies
    min/max clipping, and smooths the result.

    Args:
      runtime_params: Runtime parameters for the simulation at the current time.
      geo: Geometry of the torus at the current time.
      core_profiles: Core plasma profiles at the current time.
      pedestal_model_output: Output of the pedestal model.
      two_point_mask: Boolean mask indicating the two-point model region.
      transport_state: Transport state carried over from the previous timestep,
        containing core and pedestal component states.

    Returns:
      A tuple of (turbulent_transport, updated_transport_state), where
      turbulent_transport contains the combined 4-channel coefficients and
      individual model outputs, and updated_transport_state is the updated
      TransportState.
    """
    pedestal_mask = jnp.asarray(
        pedestal_model_output.is_active
        & pedestal_model_output_lib.pedestal_face_mask(
            geo.rho_face_norm,
            pedestal_model_output.rho_norm_ped_top,
        )
    )
    if isinstance(
        pedestal_model_output, pedestal_model_output_lib.IBCPedestalModelOutput
    ):
      core_domain_mask = jnp.where(
          pedestal_model_output.is_active,
          ~pedestal_mask,
          True,
      )
    else:
      core_domain_mask = jnp.ones_like(pedestal_mask, dtype=bool)

    core_smoothing_matrix = _build_smoothing_matrix(
        runtime_params.transport,
        geo,
        pedestal_model_output,
    )
    pedestal_smoothing_matrix = _build_pedestal_smoothing_matrix(
        runtime_params.transport,
        geo,
        pedestal_model_output,
    )

    core, core_components, new_core_states = self._compute_domain_coeffs(
        self.core_transport_models,
        runtime_params.transport.core_transport_model_params,
        runtime_params,
        geo,
        core_profiles,
        two_point_mask,
        domain_mask=core_domain_mask,
        smoothing_matrix=core_smoothing_matrix,
        component_states=transport_state.core_transport_states,
    )
    pedestal, pedestal_components, new_pedestal_states = (
        self._compute_domain_coeffs(
            self.pedestal_transport_models,
            runtime_params.transport.pedestal_transport_model_params,
            runtime_params,
            geo,
            core_profiles,
            two_point_mask,
            domain_mask=pedestal_mask,
            smoothing_matrix=pedestal_smoothing_matrix,
            component_states=transport_state.pedestal_transport_states,
        )
    )

    new_transport_state = transport_state_lib.TransportState(
        core_transport_states=new_core_states,
        pedestal_transport_states=new_pedestal_states,
    )

    return (
        transport_coeffs.TurbulentTransport(
            core=core,
            pedestal=pedestal,
            core_components=core_components,
            pedestal_components=pedestal_components,
        ),
        new_transport_state,
    )

  def _compute_domain_coeffs(
      self,
      models: Mapping[str, component.ComponentTransportModel],
      params_map: Mapping[
          str, transport_runtime_params_lib.ComponentRuntimeParams
      ],
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      domain_mask: jax.Array,
      smoothing_matrix: jax.Array,
      component_states: Mapping[
          str, transport_state_lib.ComponentTransportState
      ],
  ) -> tuple[
      transport_coeffs.TransportCoeffs,
      dict[str, transport_coeffs.TransportCoeffs],
      dict[str, transport_state_lib.ComponentTransportState],
  ]:
    """Combines, clips, masks, and smooths transport models for a domain."""
    # Note: Even when models is empty, we still apply clipping to provide a
    # baseline minimum transport floor (chi_min) in the domain. In
    # ADAPTIVE_TRANSPORT mode, this means we may have slightly more transport
    # in the pedestal region.
    combined, model_outputs, new_component_states = self._combine(
        models,
        params_map,
        runtime_params,
        geo,
        core_profiles,
        two_point_mask,
        component_states=component_states,
    )
    clipped = self._apply_clipping(runtime_params.transport, combined)
    masked = jax.tree.map(
        lambda x: jnp.where(domain_mask, x, 0.0),
        clipped,
    )
    smoothed = self._smooth_coeffs(
        masked,
        smoothing_matrix,
    )
    return smoothed, model_outputs, new_component_states

  def _combine(
      self,
      models: Mapping[str, component.ComponentTransportModel],
      params_map: Mapping[
          str, transport_runtime_params_lib.ComponentRuntimeParams
      ],
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      component_states: Mapping[
          str, transport_state_lib.ComponentTransportState
      ],
  ) -> tuple[
      transport_coeffs.TransportCoeffs,
      dict[str, transport_coeffs.TransportCoeffs],
      dict[str, transport_state_lib.ComponentTransportState],
  ]:
    """Calculates and combines transport coefficients from a dict of models."""

    # Initialize accumulators with zeros. Will be iteratively updated based on
    # model outputs and merge modes.
    zero_profile = jnp.zeros_like(
        geo.rho_face_norm, dtype=jax_utils.get_dtype()
    )
    zero_bool = jnp.zeros_like(geo.rho_face_norm, dtype=bool)
    accumulators = {}
    locks = {}

    for channel in ('chi_face_ion', 'chi_face_el', 'd_face_el', 'v_face_el'):
      accumulators[channel] = zero_profile
      locks[channel] = zero_bool
    model_outputs: dict[str, transport_coeffs.TransportCoeffs] = {}
    new_component_states: dict[
        str, transport_state_lib.ComponentTransportState
    ] = {}

    for name in models:
      model = models[name]
      params = params_map[name]
      component_state = component_states[name]
      # Calculate raw coefficients and zero out disabled channels.
      coeffs, new_component_state = model(
          params,
          runtime_params,
          geo,
          core_profiles,
          two_point_mask=two_point_mask,
          component_state=component_state,
      )

      radial_range_mask = component.compute_radial_range_mask(params, geo)

      model_outputs[name] = coeffs
      new_component_states[name] = new_component_state

      channels_and_flags = (
          ('chi_face_ion', params.disable_chi_i),
          ('chi_face_el', params.disable_chi_e),
          ('d_face_el', params.disable_D_e),
          ('v_face_el', params.disable_V_e),
      )

      for channel, is_disabled in channels_and_flags:
        # A channel is active for this model if it's in the domain AND enabled.
        channel_active = jnp.logical_and(
            radial_range_mask, jnp.logical_not(is_disabled)
        )

        val = getattr(coeffs, channel)
        if params.merge_mode == enums.MergeMode.OVERWRITE:
          # Wiping: Replace accumulator values where active.
          accumulators[channel] = jnp.where(
              channel_active, val, accumulators[channel]
          )
          # Update lock.
          locks[channel] = jnp.logical_or(locks[channel], channel_active)
        else:  # ADD
          # Add where active and not locked.
          condition = channel_active & ~locks[channel]
          accumulators[channel] = accumulators[channel] + jnp.where(
              condition, val, 0.0
          )

    return (
        transport_coeffs.TransportCoeffs(**accumulators),
        model_outputs,
        new_component_states,
    )

  def _apply_clipping(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      input_coeffs: transport_coeffs.TransportCoeffs,
  ) -> transport_coeffs.TransportCoeffs:
    """Applies min/max clipping to transport coefficients for PDE stability."""
    chi_face_ion = jnp.clip(
        input_coeffs.chi_face_ion,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    chi_face_el = jnp.clip(
        input_coeffs.chi_face_el,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    d_face_el = jnp.clip(
        input_coeffs.d_face_el,
        transport_runtime_params.D_e_min,
        transport_runtime_params.D_e_max,
    )
    v_face_el = jnp.clip(
        input_coeffs.v_face_el,
        transport_runtime_params.V_e_min,
        transport_runtime_params.V_e_max,
    )

    return transport_coeffs.TransportCoeffs(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _smooth_coeffs(
      self,
      input_coeffs: transport_coeffs.TransportCoeffs,
      smoothing_matrix: jax.Array,
  ) -> transport_coeffs.TransportCoeffs:
    """Gaussian smoothing of turbulent transport coefficients."""
    return jax.tree.map(
        lambda coeff: jnp.dot(smoothing_matrix, coeff), input_coeffs
    )


def _build_smoothing_matrix(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Builds a smoothing matrix for the transport model."""
  # 1. Build smoothing width profile
  # Spatially-varying sigma(rho) is constructed across radial grid points.
  # This supports multiple smoothing_zones with distinct widths.
  has_zones = len(transport_runtime_params.smoothing_zones) > 0

  def build_profile_from_zones():
    profile = jnp.zeros_like(geo.rho_face_norm)
    for zone in transport_runtime_params.smoothing_zones:
      in_zone = jnp.logical_and(
          geo.rho_face_norm >= zone.rho_min,
          geo.rho_face_norm <= zone.rho_max,
      )
      profile = jnp.where(in_zone, zone.smoothing_width, profile)
    return profile

  def build_profile_fallback():
    return jnp.full_like(
        geo.rho_face_norm, transport_runtime_params.smoothing_width
    )

  if has_zones:
    smoothing_width_profile = build_profile_from_zones()
  else:
    smoothing_width_profile = build_profile_fallback()

  # Apply pedestal mask if in INTERNAL_BOUNDARY_CONDITION mode.
  # Zeros out smoothing_width_profile for rho >= rho_ped_top to avoid bleeding
  # into boundary condition zone.
  if isinstance(
      pedestal_model_output, pedestal_model_output_lib.IBCPedestalModelOutput
  ):
    rho_norm_ped_top = pedestal_model_output.rho_norm_ped_top
    smoothing_width_profile = jnp.where(
        pedestal_model_output.is_active
        & pedestal_model_output_lib.pedestal_face_mask(
            geo.rho_face_norm, rho_norm_ped_top
        ),
        0.0,
        smoothing_width_profile,
    )

  return math_utils.gaussian_smoothing_matrix(
      geo.rho_face_norm,
      smoothing_width_profile,
      active=smoothing_width_profile > MIN_SMOOTHING_WIDTH,
      lower_cutoff=SMOOTHING_KERNEL_LOWER_CUTOFF,
  )


def _build_pedestal_smoothing_matrix(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Builds a smoothing matrix localized to the pedestal region."""
  rho_norm_ped_top = pedestal_model_output.rho_norm_ped_top
  smoothing_width_profile = jnp.where(
      pedestal_model_output_lib.pedestal_face_mask(
          geo.rho_face_norm, rho_norm_ped_top
      ),
      transport_runtime_params.pedestal_smoothing_width,
      0.0,
  )
  return math_utils.gaussian_smoothing_matrix(
      geo.rho_face_norm,
      smoothing_width_profile,
      active=smoothing_width_profile > MIN_SMOOTHING_WIDTH,
      lower_cutoff=SMOOTHING_KERNEL_LOWER_CUTOFF,
  )
