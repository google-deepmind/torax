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

"""Saturation model based on deviation from pedestal model."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base

# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True, eq=False)
class ProfileValueSaturationModel(base.SaturationModel):
  """Saturation model based on values of the profiles at the pedestal top."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> array_typing.FloatScalar:
    """Calculates transport increase multipliers."""
    # Get the current profile values at the pedestal top.
    # Interpolating to get the values at exactly rho_norm_ped_top is difficult,
    # as the gradients in the pedestal and the core are very different and are
    # going to be varying a lot during a solve step. Instead, we take the values
    # at the nearest grid point, which is more stable.
    rho_norm_face_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_face_norm - pedestal_output.rho_norm_ped_top)
    )
    current_T_e_ped_top = core_profiles.T_e.face_value()[
        rho_norm_face_ped_top_idx
    ]
    current_T_i_ped_top = core_profiles.T_i.face_value()[
        rho_norm_face_ped_top_idx
    ]

    # Compute the multipliers based on the deviation from the pedestal model.
    chi_e_multiplier = self._calculate_multiplier(
        current_T_e_ped_top, pedestal_output.T_e_ped, runtime_params.pedestal
    )
    chi_i_multiplier = self._calculate_multiplier(
        current_T_i_ped_top, pedestal_output.T_i_ped, runtime_params.pedestal
    )

    return pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=chi_e_multiplier,
        chi_i_multiplier=chi_i_multiplier,
        # TODO(b/487920703): set the density transport coefficients based on
        # n_e_ped. In testing, we found this could be unstable.
        D_e_multiplier=chi_e_multiplier,
        v_e_multiplier=chi_e_multiplier,
    )

  def _calculate_multiplier(
      self,
      current: array_typing.FloatScalar,
      target: array_typing.FloatScalar,
      pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
  ) -> array_typing.FloatScalar:
    """Calculates the transport increase multiplier.

    If current >> target, multiplier -> infinity.
    If current << target, multiplier -> 1.

    Args:
      current: The current value of the profile at the pedestal top.
      target: The target value of the profile at the pedestal top.
      pedestal_runtime_params: The runtime parameters for the pedestal model.

    Returns:
      The transport increase multiplier.
    """
    width = pedestal_runtime_params.saturation.sigmoid_width
    exponent = pedestal_runtime_params.saturation.sigmoid_exponent
    offset = pedestal_runtime_params.saturation.sigmoid_offset
    normalized_deviation = (current - target) / target - offset
    transport_multiplier = 1 / (
        1 - jax.nn.sigmoid(normalized_deviation / width)
    )
    transport_multiplier = transport_multiplier**exponent
    transport_multiplier = jnp.clip(
        transport_multiplier,
        min=pedestal_runtime_params.min_transport_multiplier,
        max=pedestal_runtime_params.max_transport_multiplier,
    )

    return transport_multiplier
