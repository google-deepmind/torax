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
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime params for the CGM transport model."""

  alpha: float
  chi_stiff: float
  chi_e_i_ratio: array_typing.FloatScalar
  chi_D_ratio: array_typing.FloatScalar
  VR_D_ratio: array_typing.FloatScalar


class CriticalGradientTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      transport_runtime_params: runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model.TurbulentTransport:
    r"""Calculates transport coefficients using the Critical Gradient Model.

    Uses critical normalized logarithmic ion temperature gradient
    :math:`R/L_{Ti}|_crit` from Guo Romanelli 1993:
    :math:`\chi_i = \chi_{GB} \chi_{stiff} H(R/L_{Ti} - R/L_{Ti})`
    where :math:`\chi_{GB}` is the GyroBohm diffusivity,
    :math:`\chi_{stiff}` is the stiffness parameter, and
    :math:`H` is the Heaviside function.

    Args:
      transport_runtime_params: Input runtime parameters for this transport
        model at the current time.
      runtime_params: Input runtime parameters at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """

    # pylint: disable=invalid-name

    constants = constants_module.CONSTANTS

    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    s = core_profiles.s_face
    q = core_profiles.q_face

    # very basic sawtooth model
    s = jnp.where(q < 1, 0, s)
    q = jnp.where(q < 1, 1, q)

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.R_out - geo.R_in) * 0.5

    T_i_face = core_profiles.T_i.face_value()
    T_i_face_grad = core_profiles.T_i.face_grad(rmid)
    T_e_face = core_profiles.T_e.face_value()

    # set critical gradient
    rlti_crit = (
        4.0 / 3.0 * (1.0 + T_i_face / T_e_face) * (1.0 + 2.0 * jnp.abs(s) / q)
    )

    # gyrobohm diffusivity
    chiGB = (
        (runtime_params.plasma_composition.main_ion.A_avg * constants.m_amu)
        ** 0.5
        / (constants.qe * geo.B_0) ** 2
        * (T_i_face * constants.keV2J) ** 1.5
        / geo.R_major
    )

    # R/LTi profile from current timestep T_i
    rlti = -geo.R_major * T_i_face_grad / T_i_face

    # build CGM model ion heat transport coefficient
    chi_face_ion = jnp.where(
        rlti >= rlti_crit,
        chiGB
        * transport_runtime_params.chi_stiff
        * (rlti - rlti_crit) ** transport_runtime_params.alpha,
        0.0,
    )

    # set electron heat transport coefficient to user-defined ratio of ion heat
    # transport coefficient
    chi_face_el = chi_face_ion / transport_runtime_params.chi_e_i_ratio

    # set electron particle transport coefficient to user-defined ratio of ion
    # heat transport coefficient
    d_face_el = chi_face_ion / transport_runtime_params.chi_D_ratio

    # User-provided convection coefficient
    v_face_el = d_face_el * transport_runtime_params.VR_D_ratio / geo.R_major

    return transport_model.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self):
    # All CriticalGradientModels are equivalent and can hash the same
    return hash('CriticalGradientModel')

  def __eq__(self, other):
    return isinstance(other, CriticalGradientTransportModel)
