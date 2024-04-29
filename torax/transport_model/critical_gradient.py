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

"""The CriticalGradientModel class."""

from __future__ import annotations

import chex
from jax import numpy as jnp
from torax import constants as constants_module
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass(eq=True, frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # Exponent of chi power law: chi \propto (R/LTi - R/LTi_crit)^alpha
  CGMalpha: float = 2.0
  # Stiffness parameter
  CGMchistiff: float = 2.0
  # Ratio of electron to ion transport coefficient (ion higher: ITG)
  CGMchiei_ratio: float = 2.0
  CGM_D_ratio: float = 5.0

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the CGM transport model."""

  CGMalpha: float
  CGMchistiff: float
  CGMchiei_ratio: float
  CGM_D_ratio: float

  def sanity_check(self):
    runtime_params_lib.DynamicRuntimeParams.sanity_check(self)
    # Using the object.__setattr__ call to get around the fact that this
    # dataclass is frozen.
    object.__setattr__(
        self,
        'CGM_D_ratio',
        jax_utils.error_if_negative(self.CGM_D_ratio, 'CGM_D_ratio'),
    )

  def __post_init__(self):
    self.sanity_check()


class CriticalGradientModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(
      self,
      runtime_params: RuntimeParams | None = None,
  ):
    self._runtime_params = runtime_params or RuntimeParams()

  @property
  def runtime_params(self) -> RuntimeParams:
    return self._runtime_params

  @runtime_params.setter
  def runtime_params(self, runtime_params: RuntimeParams) -> None:
    self._runtime_params = runtime_params

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    """Calculates transport coefficients using the Critical Gradient Model.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.

    Returns:
      coeffs: The transport coefficients
    """

    # Many variables throughout this function are capitalized based on physics
    # notational conventions rather than on Google Python style
    # pylint: disable=invalid-name

    # ITG critical gradient model. R/LTi_crit from Guo Romanelli 1993
    # chi_i = chiGB * chistiff * H(R/LTi -
    #  R/LTi_crit)*(R/LTi - R/LTi_crit)^alpha

    constants = constants_module.CONSTANTS
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    # set typical values for now. Will include user-defined q and s later
    s = core_profiles.s_face
    q = core_profiles.q_face

    # very basic sawtooth model
    s = jnp.where(q < 1, 0, s)
    q = jnp.where(q < 1, 1, q)

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5

    temp_ion_face = core_profiles.temp_ion.face_value()
    temp_ion_face_grad = core_profiles.temp_ion.face_grad(rmid)
    temp_el_face = core_profiles.temp_el.face_value()

    # set critical gradient
    rlti_crit = (
        4.0
        / 3.0
        * (1.0 + temp_ion_face / temp_el_face)
        * (1.0 + 2.0 * jnp.abs(s) / q)
    )

    # gyrobohm diffusivity
    chiGB = (
        (dynamic_runtime_params_slice.plasma_composition.Ai * constants.mp)
        ** 0.5
        / (constants.qe * geo.B0) ** 2
        * (temp_ion_face * constants.keV2J) ** 1.5
        / geo.Rmaj
    )

    # R/LTi profile from current timestep temp_ion
    rlti = -geo.Rmaj * temp_ion_face_grad / temp_ion_face

    # set minimum chi for PDE stability
    chi_ion = dynamic_runtime_params_slice.transport.chimin * jnp.ones_like(
        geo.mesh.face_centers
    )

    # built CGM model ion heat transport coefficient
    chi_ion = jnp.where(
        rlti >= rlti_crit,
        chiGB
        * dynamic_runtime_params_slice.transport.CGMchistiff
        * (rlti - rlti_crit) ** dynamic_runtime_params_slice.transport.CGMalpha,
        chi_ion,
    )

    # set (high) ceiling to CGM flux for PDE stability
    # (might not be necessary with Perezerev)
    chi_ion = jnp.where(
        chi_ion > dynamic_runtime_params_slice.transport.chimax,
        dynamic_runtime_params_slice.transport.chimax,
        chi_ion,
    )

    # set low transport in pedestal region to facilitate PDE solver
    # (more consistency between desired profile and transport coefficients)
    chi_face_ion = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal,
            geo.r_face_norm
            >= dynamic_runtime_params_slice.profile_conditions.Ped_top,
        ),
        dynamic_runtime_params_slice.transport.chimin,
        chi_ion,
    )

    # set electron heat transport coefficient to user-defined ratio of ion heat
    # transport coefficient
    chi_face_el = (
        chi_face_ion / dynamic_runtime_params_slice.transport.CGMchiei_ratio
    )

    d_face_el = (
        chi_face_ion / dynamic_runtime_params_slice.transport.CGM_D_ratio
    )

    # No convection in this critical gradient model.
    # (Not a realistic model for particle transport anyway).
    v_face_el = jnp.zeros_like(d_face_el)

    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
