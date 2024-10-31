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
"""Utils to handle inputs and outputs of quasilinear models."""

import chex
import jax
from jax import numpy as jnp
from torax import constants as constants_module
from torax import geometry
from torax import state
from torax.transport_model import runtime_params as runtime_params_lib


# pylint: disable=invalid-name
@chex.dataclass
class QuasilinearRuntimeParams(runtime_params_lib.RuntimeParams):
  """Shared parameters for Quasilinear models."""
  # effective D / effective V approach for particle transport
  DVeff: bool = False
  # minimum |R/Lne| below which effective V is used instead of effective D
  An_min: float = 0.05

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass(frozen=True)
class QuasilinearDynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Shared parameters for Quasilinear models."""
  DVeff: bool
  An_min: float


class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: QuasilinearRuntimeParams

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> QuasilinearDynamicRuntimeParams:
    return QuasilinearDynamicRuntimeParams(
        **self.get_dynamic_params_kwargs(t)
    )


@chex.dataclass(frozen=True)
class QuasilinearInputs:
  """Variables required to convert outputs to TORAX CoreTransport outputs."""
  chiGB: chex.Array
  Rmin: chex.Array
  Rmaj: chex.Array
  Ati: chex.Array
  Ate: chex.Array
  Ane: chex.Array


def make_core_transport(
    qi: jax.Array,
    qe: jax.Array,
    pfe: jax.Array,
    quasilinear_inputs: QuasilinearInputs,
    transport: QuasilinearDynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> state.CoreTransport:
  """Converts model output to CoreTransport."""
  constants = constants_module.CONSTANTS

  # conversion to SI units (note that n is normalized here)
  pfe_SI = (
      pfe
      * core_profiles.ne.face_value()
      * quasilinear_inputs.chiGB
      / quasilinear_inputs.Rmin
  )

  # chi outputs in SI units.
  # chi in GB units is Q[GB]/(a/LT) , Lref=Rmin in Q[GB].
  # max/min clipping included
  chi_face_ion = (
      ((quasilinear_inputs.Rmaj / quasilinear_inputs.Rmin) * qi)
      / quasilinear_inputs.Ati
  ) * quasilinear_inputs.chiGB
  chi_face_el = (
      ((quasilinear_inputs.Rmaj / quasilinear_inputs.Rmin) * qe)
      / quasilinear_inputs.Ate
  ) * quasilinear_inputs.chiGB

  # Effective D / Effective V approach.
  # For small density gradients or up-gradient transport, set pure effective
  # convection. Otherwise pure effective diffusion.
  def DVeff_approach() -> tuple[jax.Array, jax.Array]:
    # The geo.rho_b is to unnormalize the face_grad.
    Deff = -pfe_SI / (
        core_profiles.ne.face_grad() * geo.g1_over_vpr2_face * geo.rho_b
        + constants.eps
    )
    Veff = pfe_SI / (
        core_profiles.ne.face_value() * geo.g0_over_vpr_face * geo.rho_b
    )
    Deff_mask = (
        ((pfe >= 0) & (quasilinear_inputs.Ane >= 0))
        | ((pfe < 0) & (quasilinear_inputs.Ane < 0))
    ) & (abs(quasilinear_inputs.Ane) >= transport.An_min)
    Veff_mask = jnp.invert(Deff_mask)
    # Veff_mask is where to use effective V only, so zero out D there.
    d_face_el = jnp.where(Veff_mask, 0.0, Deff)
    # And vice versa
    v_face_el = jnp.where(Deff_mask, 0.0, Veff)
    return d_face_el, v_face_el

  # Scaled D approach. Scale electron diffusivity to electron heat
  # conductivity (this has some physical motivations),
  # and set convection to then match total particle transport
  def Dscaled_approach() -> tuple[jax.Array, jax.Array]:
    chex.assert_rank(pfe, 1)
    d_face_el = chi_face_el
    v_face_el = (
        pfe_SI / core_profiles.ne.face_value()
        - quasilinear_inputs.Ane
        * d_face_el
        / quasilinear_inputs.Rmaj
        * geo.g1_over_vpr2_face
        * geo.rho_b**2
    ) / (geo.g0_over_vpr_face * geo.rho_b)
    return d_face_el, v_face_el

  d_face_el, v_face_el = jax.lax.cond(
      transport.DVeff,
      DVeff_approach,
      Dscaled_approach,
  )

  return state.CoreTransport(
      chi_face_ion=chi_face_ion,
      chi_face_el=chi_face_el,
      d_face_el=d_face_el,
      v_face_el=v_face_el,
  )
