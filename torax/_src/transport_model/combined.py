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

"""The CombinedTransportModel class.

A class for combining transport models.
"""

import dataclasses
from typing import Callable, Sequence
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib

# pylint: disable=protected-access


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params_lib.RuntimeParams):
  transport_model_params: Sequence[transport_runtime_params_lib.RuntimeParams]
  pedestal_transport_model_params: Sequence[
      transport_runtime_params_lib.RuntimeParams
  ]


@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportModel(transport_model_lib.TransportModel):
  """Combines coefficients from a tuple of transport models."""

  transport_models: tuple[transport_model_lib.TransportModel, ...]
  pedestal_transport_models: tuple[transport_model_lib.TransportModel, ...]

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:

    transport_runtime_params = runtime_params.transport

    # Calculate the transport coefficients - includes contribution from pedestal
    # and core transport models.
    transport_coeffs = self.call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    )

    # In contrast to the base TransportModel, we do not apply domain restriction
    # or output masking (enabled/disabled channels) as these are handled at the
    # component model level in call_implementation here.

    # Apply min/max clipping
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )

    # In contrast to the base TransportModel, we do not apply patches, as these
    # should be handled by instantiating constant component models instead.
    # However, the rho_inner and rho_outer arguments are currently required
    # in the case where the inner/outer region are to be excluded from
    # smoothing. Smoothing is applied to
    # rho_inner < rho_norm < min(rho_ped_top, rho_outer) unless
    # smooth_everywhere is True.
    return self._smooth_coeffs(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Combined model.

    Args:
      transport_runtime_params: Input runtime parameters for this transport
        model. Can change without triggering a JAX recompilation.
      runtime_params: Runtime parameters for the simulation at the current time.
      geo: Geometry of the torus at the current time.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    core_coeffs = self._combine(
        self.transport_models,
        transport_runtime_params.transport_model_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
        transport_model_lib.compute_core_domain_mask,
    )

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
    combined_transport_coeffs = jax.tree.map(
        _add_optional, core_coeffs, pedestal_coeffs
    )

    return combined_transport_coeffs

  def _combine(
      self,
      models: tuple[transport_model_lib.TransportModel, ...],
      params_list: Sequence[transport_runtime_params_lib.RuntimeParams],
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
      domain_mask_fn: Callable[
          [
              transport_runtime_params_lib.RuntimeParams,
              geometry.Geometry,
              pedestal_model_lib.PedestalModelOutput,
          ],
          jax.Array,
      ],
  ) -> transport_model_lib.TurbulentTransport:
    """Calculates and combines transport coefficients from a list of models."""

    # Initialize accumulators with zeros. Will be iteratively updated based on
    # model outputs and merge modes.
    zero_profile = jnp.zeros_like(
        geo.rho_face_norm, dtype=jax_utils.get_dtype()
    )
    accumulators = {}
    locks = {}

    for channel, config in transport_model_lib.CHANNEL_CONFIG_STRUCT.items():
      accumulators[channel] = zero_profile
      locks[channel] = jnp.zeros_like(geo.rho_face_norm, dtype=bool)
      for sub in config['sub_channels']:
        accumulators[sub] = None

    # TODO(b/344023668) explore batching or fori_loop for performance.
    for model, params in zip(models, params_list, strict=True):
      # 1. Calculate raw coefficients
      coeffs = model.call_implementation(
          params, runtime_params, geo, core_profiles, pedestal_model_output
      )

      # 2. Zero out disabled channels. Unused subchannels returned as None.
      coeffs = model.zero_out_disabled_channels(params, coeffs)

      # 3. Calculate active domain mask. Values outside this are set to 0.
      domain_mask = domain_mask_fn(params, geo, pedestal_model_output)

      coeffs_dict = dataclasses.asdict(coeffs)
      for k in coeffs_dict:
        # Apply domain restriction to values.
        if coeffs_dict[k] is not None:
          coeffs_dict[k] = jnp.where(domain_mask, coeffs_dict[k], 0.0)

      for channel, config in transport_model_lib.CHANNEL_CONFIG_STRUCT.items():
        disable_flag_name = config['disable_flag']
        is_disabled = getattr(params, disable_flag_name)

        # A channel is active for this model if it's in the domain AND enabled.
        # Note that this is a boolean array over the face grid.
        channel_active = jnp.logical_and(
            domain_mask, jnp.logical_not(is_disabled)
        )

        val = coeffs_dict[channel]
        if params.merge_mode == enums.MergeMode.OVERWRITE:
          # Wiping: Replace accumulator values where active.
          accumulators[channel] = jnp.where(
              channel_active, val, accumulators[channel]
          )
          # Update lock.
          locks[channel] = jnp.logical_or(locks[channel], channel_active)
        else:  # ADD
          # Add where not locked.
          factor = jnp.where(locks[channel], 0.0, 1.0)
          accumulators[channel] = accumulators[channel] + val * factor

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

    return transport_model_lib.TurbulentTransport(**accumulators)


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
    unused_params: transport_runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_output: pedestal_model_lib.PedestalModelOutput,
) -> jax.Array:
  """Calculates the active domain mask for pedestal transport models."""
  return jnp.asarray(geo.rho_face_norm > pedestal_output.rho_norm_ped_top)
