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

from jax import numpy as jnp
from torax import config_slice
from torax import constants as constants_module
from torax import geometry
from torax import state
from torax.transport_model import transport_model


class CriticalGradientModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def _call_implementation(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_model.TransportCoeffs:
    """Calculates transport coefficients using the Critical Gradient Model.

    Args:
      dynamic_config_slice: Input config parameters that can change without
        triggering a JAX recompilation.
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
        (dynamic_config_slice.Ai * constants.mp) ** 0.5
        / (constants.qe * geo.B0) ** 2
        * (temp_ion_face * constants.keV2J) ** 1.5
        / dynamic_config_slice.Rmaj
    )

    # R/LTi profile from current timestep temp_ion
    rlti = -dynamic_config_slice.Rmaj * temp_ion_face_grad / temp_ion_face

    # set minimum chi for PDE stability
    chi_ion = dynamic_config_slice.transport.chimin * jnp.ones_like(
        geo.mesh.face_centers
    )

    # built CGM model ion heat transport coefficient
    chi_ion = jnp.where(
        rlti >= rlti_crit,
        chiGB
        * dynamic_config_slice.transport.CGMchistiff
        * (rlti - rlti_crit) ** dynamic_config_slice.transport.CGMalpha,
        chi_ion,
    )

    # set (high) ceiling to CGM flux for PDE stability
    # (might not be necessary with Perezerev)
    chi_ion = jnp.where(
        chi_ion > dynamic_config_slice.transport.chimax,
        dynamic_config_slice.transport.chimax,
        chi_ion,
    )

    # set low transport in pedestal region to facilitate PDE solver
    # (more consistency between desired profile and transport coefficients)
    chi_face_ion = jnp.where(
        jnp.logical_and(
            dynamic_config_slice.set_pedestal,
            geo.r_face_norm >= dynamic_config_slice.Ped_top,
        ),
        dynamic_config_slice.transport.chimin,
        chi_ion,
    )

    # set electron heat transport coefficient to user-defined ratio of ion heat
    # transport coefficient
    chi_face_el = chi_face_ion / dynamic_config_slice.transport.CGMchiei_ratio

    d_face_el = chi_face_ion / dynamic_config_slice.transport.CGM_D_ratio

    # No convection in this critical gradient model.
    # (Not a realistic model for particle transport anyway).
    v_face_el = jnp.zeros_like(d_face_el)

    return transport_model.TransportCoeffs(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
