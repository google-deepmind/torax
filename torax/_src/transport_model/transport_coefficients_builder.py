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

import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import pereverzev as pereverzev_lib
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib
from torax._src.transport_model import transport_model as transport_model_lib

# pylint: disable=invalid-name


def _compute_two_point_face_mask(
    geo: geometry.Geometry,
    runtime_params: runtime_params_lib.RuntimeParams,
    pedestal_model_output: (
        pedestal_model_output_lib.PedestalModelOutput | None
    ) = None,
) -> array_typing.BoolVectorFace:
  """Computes a boolean mask for faces that should use 2-point central differencing.

  Combines 2-point face masks from both pedestal and internal boundary
  conditions.

  Args:
    geo: Geometry of the torus.
    runtime_params: Runtime parameters for the simulation.
    pedestal_model_output: Output of the pedestal model.

  Returns:
    A boolean array on the face grid indicating which faces should use 2-point
    central differencing.
  """
  mask = jnp.zeros_like(geo.rho_face_norm, dtype=bool)
  if pedestal_model_output is not None:
    mask = mask | pedestal_model_output.get_two_point_face_mask(
        geo, set_pedestal=runtime_params.pedestal.set_pedestal
    )
  if runtime_params.profile_conditions.internal_boundary_conditions is not None:
    ibc = runtime_params.profile_conditions.internal_boundary_conditions
    mask = mask | ibc.get_two_point_face_mask(geo)
  return mask


@jax.jit(
    static_argnames=(
        'transport_model',
        'neoclassical_models',
    )
)
def calculate_all_transport_coeffs(
    transport_model: transport_model_lib.TransportModel,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    use_pereverzev: bool = False,
) -> state.CoreTransport:
  """Calculates the transport coefficients from all models."""

  # Toggle the pedestal model on/off based on the pedestal transition state.
  # TODO(b/434175938): Find an alternative method for propagating pedestal
  # transition state to the core transport masking. Currently, we're overriding
  # the runtime params which is a bit hacky. Options include passing the
  # transition state to the pedestal model or to the transport model, both of
  # which are breaking API changes.
  if (
      runtime_params.pedestal.use_formation_model_with_internal_boundary_condition
  ):
    # Pedestal model is active if we are in H-mode or in a transition.
    set_pedestal = (
        pedestal_transition_state.confinement_mode
        != pedestal_transition_state_lib.ConfinementMode.L_MODE
    )

    pedestal_params = dataclasses.replace(
        runtime_params.pedestal,
        set_pedestal=set_pedestal,
    )
    runtime_params = dataclasses.replace(
        runtime_params,
        pedestal=pedestal_params,
    )

  pedestal_model_output = pedestal_transition_state.pedestal_model_output
  two_point_mask = _compute_two_point_face_mask(
      geo=geo,
      runtime_params=runtime_params,
      pedestal_model_output=pedestal_model_output,
  )
  turbulent_transport_coeffs = transport_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_model_output=pedestal_model_output,
      two_point_mask=two_point_mask,
  )
  neoclassical_transport_coeffs = neoclassical_models.transport(
      runtime_params,
      geo,
      core_profiles,
  )

  # TODO(b/311653933) this pattern for Pereverzev-Corrigan terms forces us to
  # include value zero convection terms in the discrete system, slowing
  # compilation down by ~10%. See if can improve with a different pattern.
  # TODO(b/485528848) Replace cond with if.
  pereverzev_transport_coeffs = jax.lax.cond(
      use_pereverzev,
      pereverzev_lib.calculate_pereverzev_transport,
      lambda *_: transport_coeffs_lib.PereverzevTransport.zeros(geo),
      runtime_params,
      geo,
      core_profiles,
      two_point_mask,
  )

  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    # If in INTERNAL_BOUNDARY_CONDITION mode, set the Pereverzev transport
    # coefficients in the pedestal region to zero.
    # TODO(b/485147781) Combine this masking with the turbulent transport
    # masking.
    pedestal_active_mask_face = (
        geo.rho_face_norm >= pedestal_model_output.rho_norm_ped_top
    )
    pereverzev_transport_coeffs = jax.tree_util.tree_map(
        lambda x: jnp.where(pedestal_active_mask_face, 0.0, x),
        pereverzev_transport_coeffs,
    )

  total = transport_coeffs_lib.sum_transport_coeffs(
      turbulent_transport_coeffs.total,
      neoclassical_transport_coeffs,
      pereverzev_transport_coeffs,
  )

  core_transport = state.CoreTransport(
      total=total,
      turbulent=turbulent_transport_coeffs,
      neoclassical=neoclassical_transport_coeffs,
      pereverzev=pereverzev_transport_coeffs,
  )

  # Modify the turbulent + Pereverzev transport coefficients if the pedestal
  # model is in ADAPTIVE_TRANSPORT mode.
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
  ):
    core_transport = pedestal_model_output.modify_core_transport(
        core_transport=core_transport,
        geo=geo,
        pedestal_runtime_params=runtime_params.pedestal,
    )

  return core_transport
