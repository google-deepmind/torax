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
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import pereverzev as pereverzev_lib
from torax._src.transport_model import transport_model as transport_model_lib


@jax.jit(
    static_argnames=(
        'pedestal_model',
        'transport_model',
        'neoclassical_models',
    )
)
def calculate_all_transport_coeffs(
    pedestal_model: pedestal_model_lib.PedestalModel,
    transport_model: transport_model_lib.TransportModel,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    use_pereverzev: bool = False,
) -> state.CoreTransport:
  """Calculates the transport coefficients, combining pedestal, turbulent, Pereverzev, and neoclassical models."""
  pedestal_model_output = pedestal_model(runtime_params, geo, core_profiles)
  turbulent_transport_coeffs = transport_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_model_output=pedestal_model_output,
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
      lambda runtime_params, geo, core_profiles: pereverzev_lib.PereverzevTransport.zeros(
          geo
      ),
      runtime_params,
      geo,
      core_profiles,
  )

  core_transport = state.CoreTransport(
      **dataclasses.asdict(turbulent_transport_coeffs),
      **dataclasses.asdict(neoclassical_transport_coeffs),
      **dataclasses.asdict(pereverzev_transport_coeffs),
  )

  # Modify the turbulent + Pereverzev transport coefficients if the pedestal
  # model is in ADAPTIVE_TRANSPORT mode; otherwise, set them to zero in the
  # pedestal region.
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
  ):
    assert isinstance(
        pedestal_model_output,
        pedestal_model_lib.AdaptiveTransportPedestalModelOutput,
    )
    core_transport = pedestal_model_output.modify_core_transport(
        core_transport=core_transport,
        geo=geo,
    )
  else:
    pedestal_active_mask_face = (
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top
    )
    pereverzev_transport_coeffs = jax.tree_util.tree_map(
        lambda x: jnp.where(pedestal_active_mask_face, 0.0, x),
        pereverzev_transport_coeffs,
    )
    core_transport = dataclasses.replace(
        core_transport,
        **dataclasses.asdict(pereverzev_transport_coeffs),
    )

  return core_transport
