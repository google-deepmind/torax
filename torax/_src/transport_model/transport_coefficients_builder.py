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
import functools

from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import transport_model as transport_model_lib


@functools.partial(jax_utils.jit, static_argnums=(0, 1, 2))
def calculate_total_transport_coeffs(
    pedestal_model: pedestal_model_lib.PedestalModel,
    transport_model: transport_model_lib.TransportModel,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreTransport:
  """Calculates the transport coefficients."""
  pedestal_model_output = pedestal_model(
      dynamic_runtime_params_slice_t, geo_t, core_profiles_t
  )
  turbulent_transport = transport_model(
      dynamic_runtime_params_slice_t,
      geo_t,
      core_profiles_t,
      pedestal_model_output,
  )
  neoclassical_transport_coeffs = (
      neoclassical_models.transport.calculate_neoclassical_transport(
          dynamic_runtime_params_slice_t,
          geo_t,
          core_profiles_t,
      )
  )

  return state.CoreTransport(
      **turbulent_transport,
      **neoclassical_transport_coeffs,
  )

