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

"""Code to build the combined transport coefficients for a simulation."""

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.transport_model import pereverzev as pereverzev_lib
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib
from torax._src.transport_model import transport_model as transport_model_lib
from torax._src.transport_model import transport_state as transport_state_lib

# pylint: disable=invalid-name


@jax.jit(
    static_argnames=(
        'transport_model',
        'use_pereverzev',
    ),
    inline=jax.Inline.XLA_LATE,
)
def calculate_all_transport_coeffs(
    transport_model: transport_model_lib.TransportModel,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
    transport_state: transport_state_lib.TransportState,
    neoclassical_transport: transport_coeffs_lib.NeoclassicalTransport,
    two_point_mask: array_typing.BoolVectorFace,
    use_pereverzev: bool = False,
) -> tuple[state.CoreTransport, transport_state_lib.TransportState]:
  """Calculates the transport coefficients from all models and updated state.

  Args:
    transport_model: Combined transport model.
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles.
    pedestal_model_output: Output of the pedestal model for this step.
    transport_state: Transport state carried over from the previous timestep.
    neoclassical_transport: Neoclassical transport coefficients.
    two_point_mask: Mask for boundary faces.
    use_pereverzev: Whether to calculate Pereverzev transport.

  Returns:
    A tuple of (core_transport, updated_transport_state).
  """
  turbulent_transport_coeffs, transport_state = transport_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_model_output=pedestal_model_output,
      two_point_mask=two_point_mask,
      transport_state=transport_state,
  )

  if use_pereverzev:
    pereverzev_transport_coeffs = pereverzev_lib.calculate_pereverzev_transport(
        runtime_params,
        geo,
        core_profiles,
    )
    # If in INTERNAL_BOUNDARY_CONDITION mode, set the Pereverzev transport
    # coefficients in the pedestal region to zero.
    if isinstance(
        pedestal_model_output, pedestal_model_output_lib.IBCPedestalModelOutput
    ):
      pedestal_active_mask_face = jnp.where(
          pedestal_model_output.is_active,
          geo.rho_face_norm >= pedestal_model_output.rho_norm_ped_top,
          False,
      )
      pereverzev_transport_coeffs = jax.tree.map(
          lambda x: jnp.where(pedestal_active_mask_face, 0.0, x),
          pereverzev_transport_coeffs,
      )
  else:
    pereverzev_transport_coeffs = None

  coeffs_to_sum = [
      turbulent_transport_coeffs.total,
      neoclassical_transport,
  ]
  if pereverzev_transport_coeffs is not None:
    coeffs_to_sum.append(pereverzev_transport_coeffs)
  total = transport_coeffs_lib.sum_transport_coeffs(*coeffs_to_sum)

  core_transport = state.CoreTransport(
      total=total,
      turbulent=turbulent_transport_coeffs,
      neoclassical=neoclassical_transport,
      pereverzev=pereverzev_transport_coeffs,
  )

  if isinstance(
      pedestal_model_output,
      pedestal_model_output_lib.AdaptiveTransportPedestalModelOutput,
  ):
    core_transport = pedestal_model_output.modify_core_transport(
        core_transport=core_transport,
        geo=geo,
        pedestal_runtime_params=runtime_params.pedestal,
    )

  return core_transport, transport_state
