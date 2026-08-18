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
from typing import Callable, Mapping
import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import component
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params as transport_runtime_params_lib

MIN_SMOOTHING_WIDTH = 1e-5


@dataclasses.dataclass(frozen=True, eq=False)
class TransportModel(static_dataclass.StaticDataclass):
  """Combines coefficients from a dict of named transport models."""

  core_transport_models: Mapping[str, component.ComponentTransportModel]
  pedestal_transport_models: Mapping[str, component.ComponentTransportModel]

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> component.TurbulentTransport:
    r"""Calculates transport coefficients using the TransportModel.

    Combines coefficients from core and pedestal transport models, applies
    min/max clipping, and smooths the result.

    Args:
      runtime_params: Runtime parameters for the simulation at the current time.
      geo: Geometry of the torus at the current time.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    transport_runtime_params = runtime_params.transport

    # Calculate transport coefficients from core models.
    core_coeffs = self._combine(
        self.core_transport_models,
        transport_runtime_params.core_transport_model_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
        component.compute_core_domain_mask,
    )

    # Calculate transport coefficients from pedestal models.
    pedestal_coeffs = self._combine(
        self.pedestal_transport_models,
        transport_runtime_params.pedestal_transport_model_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
        _pedestal_domain_mask,
    )

    # Combine the transport coefficients from core and pedestal models.
    transport_coeffs = jax.tree.map(
        _add_optional, core_coeffs, pedestal_coeffs
    )

    # Apply min/max clipping.
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )

    # Apply smoothing.
    transport_coeffs = self._smooth_coeffs(
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

    return transport_coeffs

  def _combine(
      self,
      models: Mapping[str, component.ComponentTransportModel],
      params_map: Mapping[
          str, transport_runtime_params_lib.ComponentRuntimeParams
      ],
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
      domain_mask_fn: Callable[
          [
              transport_runtime_params_lib.ComponentRuntimeParams,
              runtime_params_lib.RuntimeParams,
              geometry.Geometry,
              pedestal_model_output_lib.PedestalModelOutput,
          ],
          jax.Array,
      ],
  ) -> component.TurbulentTransport:
    """Calculates and combines transport coefficients from a dict of models."""

    # Initialize accumulators with zeros. Will be iteratively updated based on
    # model outputs and merge modes.
    zero_profile = jnp.zeros_like(
        geo.rho_face_norm, dtype=jax_utils.get_dtype()
    )
    accumulators = {}
    locks = {}

    for (
        channel,
        config,
    ) in component.ComponentTransportModel.CHANNEL_CONFIG.items():
      accumulators[channel] = zero_profile
      locks[channel] = jnp.zeros_like(geo.rho_face_norm, dtype=bool)
      for sub in config['sub_channels']:
        accumulators[sub] = None

    # TODO(b/344023668) explore batching or fori_loop for performance.
    for name in models:
      model = models[name]
      params = params_map[name]
      # 1. Calculate raw coefficients and zero out disabled channels.
      coeffs = model(
          params, runtime_params, geo, core_profiles, pedestal_model_output
      )

      # 2. Calculate active domain mask. Values outside this are set to 0.
      domain_mask = domain_mask_fn(
          params, runtime_params, geo, pedestal_model_output
      )

      coeffs_dict = dataclasses.asdict(coeffs)
      for k in coeffs_dict:
        # Apply domain restriction to values.
        if coeffs_dict[k] is not None:
          coeffs_dict[k] = jnp.where(domain_mask, coeffs_dict[k], 0.0)  # pyrefly: ignore[bad-argument-type]

      for (
          channel,
          config,
      ) in component.ComponentTransportModel.CHANNEL_CONFIG.items():
        disable_flag_name = config['disable_flag']
        is_disabled = getattr(params, disable_flag_name)  # pyrefly: ignore[bad-argument-type]

        # A channel is active for this model if it's in the domain AND enabled.
        # Note that this is a boolean array over the face grid.
        channel_active = jnp.logical_and(
            domain_mask, jnp.logical_not(is_disabled)
        )

        val = coeffs_dict[channel]
        if params.merge_mode == enums.MergeMode.OVERWRITE:
          # Wiping: Replace accumulator values where active.
          accumulators[channel] = jnp.where(
              channel_active, val, accumulators[channel]  # pyrefly: ignore[bad-argument-type]
          )
          # Update lock.
          locks[channel] = jnp.logical_or(locks[channel], channel_active)
        else:  # ADD
          # Add where not locked.
          factor = jnp.where(locks[channel], 0.0, 1.0)
          accumulators[channel] = accumulators[channel] + val * factor  # pyrefly: ignore[unsupported-operation]

        # Handle sub-channels.
        for sub in config['sub_channels']:
          sub_val = coeffs_dict[sub]
          if sub_val is not None:
            if accumulators[sub] is None:
              accumulators[sub] = zero_profile

            if params.merge_mode == enums.MergeMode.OVERWRITE:
              accumulators[sub] = jnp.where(
                  channel_active, sub_val, accumulators[sub]
              )
            else:  # ADD
              # Add where not locked (using main channel lock).
              factor = jnp.where(locks[channel], 0.0, 1.0)
              accumulators[sub] = accumulators[sub] + sub_val * factor

    return component.TurbulentTransport(**accumulators)

  def _apply_clipping(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      transport_coeffs: component.TurbulentTransport,
  ) -> component.TurbulentTransport:
    """Applies min/max clipping to transport coefficients for PDE stability."""
    chi_face_ion = jnp.clip(
        transport_coeffs.chi_face_ion,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    chi_face_el = jnp.clip(
        transport_coeffs.chi_face_el,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    d_face_el = jnp.clip(
        transport_coeffs.d_face_el,
        transport_runtime_params.D_e_min,
        transport_runtime_params.D_e_max,
    )
    v_face_el = jnp.clip(
        transport_coeffs.v_face_el,
        transport_runtime_params.V_e_min,
        transport_runtime_params.V_e_max,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _smooth_coeffs(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: component.TurbulentTransport,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> component.TurbulentTransport:
    """Gaussian smoothing of turbulent transport coefficients."""
    smoothing_matrix = _build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        pedestal_model_output,
    )

    # Iterate over fields of the CoreTransport dataclass.
    # Ignore optional fields that are made all zero in post_init.
    def smooth_single_coeff(coeff):
      return jax.lax.cond(
          jnp.all(coeff == 0.0),
          lambda: coeff,
          lambda: jnp.dot(smoothing_matrix, coeff),
      )

    return jax.tree.map(smooth_single_coeff, transport_coeffs)


def _add_optional(
    core_value: jax.Array | None, pedestal_value: jax.Array | None
) -> jax.Array | None:
  """Adds two values, treating None as zero. Returns None if both are None."""
  if core_value is None:
    return pedestal_value
  if pedestal_value is None:
    return core_value
  return core_value + pedestal_value


def _pedestal_domain_mask(
    unused_transport_runtime_params: (
        transport_runtime_params_lib.ComponentRuntimeParams
    ),
    unused_runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Calculates the active domain mask for pedestal transport models."""
  return jnp.asarray(geo.rho_face_norm >= pedestal_output.rho_norm_ped_top)


def _build_smoothing_matrix(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Builds a smoothing matrix for the transport model."""
  # To reduce the range of the convolution, weights under lower_cutoff are
  # clipped to zero.
  lower_cutoff = 0.01
  # used for eps, small number to avoid divisions by zero for sigma = 0
  consts = constants.CONSTANTS

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

  smoothing_width_profile = jax.lax.cond(
      has_zones,
      build_profile_from_zones,
      build_profile_fallback,
  )

  # Apply pedestal mask if in INTERNAL_BOUNDARY_CONDITION mode.
  # Zeros out smoothing_width_profile for rho >= rho_ped_top to avoid bleeding
  # into boundary condition zone.
  is_internal_boundary_condition = (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  )
  if is_internal_boundary_condition:

    def apply_pedestal_mask(profile):
      return jnp.where(
          geo.rho_face_norm < pedestal_model_output.rho_norm_ped_top,
          profile,
          0.0,
      )

    smoothing_width_profile = jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        apply_pedestal_mask,
        lambda p: p,
        smoothing_width_profile,
    )

  # 2. Kernel matrix with variable width (sigma_i for each destination row i)
  r_diff = geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm
  sigma = smoothing_width_profile[:, jnp.newaxis]

  kernel = jnp.exp(-jnp.log(2) * r_diff**2 / (sigma**2 + consts.eps))

  # 3. Dynamic active mask derivation (sigma > threshold)
  # Derives a binary active mask vector directly from smoothing_width_profile.
  mask = jnp.where(smoothing_width_profile > MIN_SMOOTHING_WIDTH, 1.0, 0.0)

  # Zero out rows (destinations) that should not be smoothed
  diag_mask = jnp.diag(mask)
  kernel = jnp.dot(diag_mask, kernel)

  # Zero out columns (sources) that should not contribute to smoothing
  num_rows = len(mask)
  mask_mat = jnp.tile(mask, (num_rows, 1))
  kernel *= mask_mat

  # Restore identity to the zero rows (so smoothing is a no-op there)
  zero_row_mask = jnp.all(kernel == 0, axis=1)
  kernel = jnp.where(
      zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]), kernel
  )

  # 4. Normalization
  row_sums = jnp.sum(kernel, axis=1, keepdims=True)
  kernel = kernel / row_sums

  # 5. Remove small numbers
  kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)

  # 6. Final Normalization following removal of small numbers
  row_sums = jnp.sum(kernel, axis=1, keepdims=True)
  kernel = kernel / row_sums

  return kernel


# TODO(b/426132633): Remove backwards compatibility alias.
CombinedTransportModel = TransportModel
