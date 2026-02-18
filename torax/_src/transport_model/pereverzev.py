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

import dataclasses

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PereverzevTransport:
  """Pereverzev-Corrigan transport coefficients."""

  chi_face_ion_pereverzev: jax.Array
  chi_face_el_pereverzev: jax.Array
  full_v_heat_face_ion_pereverzev: jax.Array
  full_v_heat_face_el_pereverzev: jax.Array
  d_face_el_pereverzev: jax.Array
  v_face_el_pereverzev: jax.Array

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> 'PereverzevTransport':
    """Returns a PereverzevTransport with all zeros."""
    return cls(
        chi_face_ion_pereverzev=jnp.zeros_like(geo.rho_face),
        chi_face_el_pereverzev=jnp.zeros_like(geo.rho_face),
        full_v_heat_face_ion_pereverzev=jnp.zeros_like(geo.rho_face),
        full_v_heat_face_el_pereverzev=jnp.zeros_like(geo.rho_face),
        d_face_el_pereverzev=jnp.zeros_like(geo.rho_face),
        v_face_el_pereverzev=jnp.zeros_like(geo.rho_face),
    )


def calculate_pereverzev_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> PereverzevTransport:
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
  full_v_heat_face_per_ion = (
      core_profiles.T_i.face_grad()
      / core_profiles.T_i.face_value()
      * geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )
  full_v_heat_face_per_el = (
      core_profiles.T_e.face_grad()
      / core_profiles.T_e.face_value()
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
      * core_profiles.n_e.face_grad()
      / core_profiles.n_e.face_value()
      * runtime_params.solver.D_pereverzev
  )

  return PereverzevTransport(
      chi_face_ion_pereverzev=chi_face_per_ion,
      chi_face_el_pereverzev=chi_face_per_el,
      full_v_heat_face_ion_pereverzev=full_v_heat_face_per_ion,
      full_v_heat_face_el_pereverzev=full_v_heat_face_per_el,
      d_face_el_pereverzev=d_face_per_el,
      v_face_el_pereverzev=v_face_per_el,
  )
