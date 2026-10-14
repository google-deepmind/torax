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

"""Pereverzev-Corrigan transport coefficients.

The Pereverzev-Corrigan terms are added to the transport coefficients to help
stabilize the linear solver when using highly nonlinear (stiff) transport
coefficients. See: G.V. Pereverzev and G. Corrigan, "Stable numeric scheme for
diffusion equation with a stiff transport", Computer Physics Communications 179
(2008) 579–585. https://doi.org/10.1016/j.cpc.2008.05.006
"""

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.transport_model import transport_coeffs


def _midpoint_face_value(var: cell_variable.CellVariable) -> jax.Array:
  """Arithmetic cell-midpoint face values matching zero-Péclet convection.

  Pereverzev counter-convection is built as `v_per = d_per * grad(u) / u_face`
  and then applied by `make_convection_terms` at zero Péclet. Discrete
  cancellation against the Pereverzev diffusion term is exact only if `u_face`
  here equals the face value that operator uses: the arithmetic mean
  `0.5 * (u[j-1] + u[j])` on interior faces, and the ghost-cell boundary values
  (`left_face_value` / `right_face_value`) at the domain edges. This differs
  from `var.face_value()`, which interpolates linearly in distance on
  non-uniform meshes.

  Args:
    var: Cell variable on the radial mesh.

  Returns:
    Array of face values with shape `(num_faces,)`.
  """
  return jnp.concatenate([
      var.left_face_value,
      0.5 * (var.value[:-1] + var.value[1:]),
      var.right_face_value,
  ])


def calculate_pereverzev_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> transport_coeffs.PereverzevTransport:
  """Calculates Pereverzev-Corrigan transport coefficients.

  Pereverzev-Corrigan adds additional transport to help deal with stiff
  nonlinearities.

  Args:
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles.

  Returns:
    Pereverzev-Corrigan transport coefficients.
  """

  consts = constants.CONSTANTS
  all_two_point = jnp.ones_like(geo.rho_face_norm, dtype=bool)

  # Heat diffusion coefficients
  chi_face_per_ion = (
      jnp.ones_like(geo.rho_face) * runtime_params.solver.chi_pereverzev
  )
  chi_face_per_el = (
      jnp.ones_like(geo.rho_face) * runtime_params.solver.chi_pereverzev
  )

  # Heat convection coefficients
  # Set such that the added heat diffusion is zeroed out by the convection terms
  # We use g1_over_vpr, rather than g1_over_vpr / g0, as we would have to later
  # multiply by g0 to get the real convection coefficient. Hence, these are
  # labeled as "full" coefficients.
  # Uses 2-point face gradients (`all_two_point`) and arithmetic cell-midpoint
  # face averages (`_midpoint_face_value`) to match `diffusion_terms` and
  # zero-Péclet `convection_terms` on non-uniform meshes.
  full_v_heat_face_per_ion = (
      core_profiles.T_i.face_grad(two_point_mask=all_two_point)
      / _midpoint_face_value(core_profiles.T_i)
      * geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )
  full_v_heat_face_per_el = (
      core_profiles.T_e.face_grad(two_point_mask=all_two_point)
      / _midpoint_face_value(core_profiles.T_e)
      * geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  # Particle diffusion coefficient
  d_face_per_el = (
      jnp.ones_like(geo.rho_face) * runtime_params.solver.D_pereverzev
  )
  g1_over_vpr_g0 = jnp.concatenate(
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  # Particle convection coefficient
  # Set such that the added particle diffusion is zeroed out by the convection
  # term
  v_face_per_el = (
      g1_over_vpr_g0
      * core_profiles.n_e.face_grad(two_point_mask=all_two_point)
      / _midpoint_face_value(core_profiles.n_e)
      * runtime_params.solver.D_pereverzev
  )

  return transport_coeffs.PereverzevTransport(
      chi_face_ion=chi_face_per_ion,
      chi_face_el=chi_face_per_el,
      d_face_el=d_face_per_el,
      v_face_el=v_face_per_el,
      full_v_heat_face_ion=full_v_heat_face_per_ion,
      full_v_heat_face_el=full_v_heat_face_per_el,
  )
