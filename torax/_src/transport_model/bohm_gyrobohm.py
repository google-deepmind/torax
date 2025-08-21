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

"""The BohmGyroBohmModel class."""
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
from torax._src.transport_model import transport_model as transport_model_lib


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  chi_e_bohm_coeff: array_typing.FloatScalar
  chi_e_gyrobohm_coeff: array_typing.FloatScalar
  chi_i_bohm_coeff: array_typing.FloatScalar
  chi_i_gyrobohm_coeff: array_typing.FloatScalar
  D_face_c1: array_typing.FloatScalar
  D_face_c2: array_typing.FloatScalar
  V_face_coeff: array_typing.FloatScalar
  chi_e_bohm_multiplier: array_typing.FloatScalar
  chi_e_gyrobohm_multiplier: array_typing.FloatScalar
  chi_i_bohm_multiplier: array_typing.FloatScalar
  chi_i_gyrobohm_multiplier: array_typing.FloatScalar


class BohmGyroBohmTransportModel(transport_model_lib.TransportModel):
  """Calculates various coefficients related to particle transport according to the Bohm + gyro-Bohm Model."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the BohmGyroBohm model.

    We use the implementation from Tholerus et al, Section 3.3.
    https://doi.org/10.1088/1741-4326/ad6ea2

    A description is provided in physics_models.rst.
    https://torax.readthedocs.io/en/latest/physics_models.html

    Args:
      transport_dynamic_runtime_params: Input runtime parameters for this
        transport model. Can change without triggering a JAX recompilation.
      dynamic_runtime_params_slice: Input runtime parameters for all components
        of the simulation that can change without triggering a JAX
        recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    del pedestal_model_output  # Unused.
    # pylint: disable=invalid-name
    # Required for pytype
    assert isinstance(transport_dynamic_runtime_params, DynamicRuntimeParams)

    # Bohm term of heat transport
    chi_e_B = (
        geo.r_mid_face
        * core_profiles.q_face**2
        / (
            constants_module.CONSTANTS.qe
            * geo.B_0
            * core_profiles.n_e.face_value()
        )
        * (
            jnp.abs(core_profiles.n_e.face_grad())
            * core_profiles.T_e.face_value()
            + jnp.abs(core_profiles.T_e.face_grad())
            * core_profiles.n_e.face_value()
        )
        * constants_module.CONSTANTS.keV2J
        / geo.rho_b
    )

    # Set proportionality of chi_i to chi_e according to the assumptions of the
    # Bohm model.
    chi_i_B = 2 * chi_e_B

    # Gyrobohm term of heat transport
    chi_e_gB = (
        jnp.sqrt(
            dynamic_runtime_params_slice.plasma_composition.main_ion.A_avg / 2
        )
        * jnp.sqrt(core_profiles.T_e.face_value() * 1e3)
        / geo.B_0**2
        * jnp.abs(core_profiles.T_e.face_grad() * 1e3)
        / geo.rho_b
    )

    # Set proportionality of chi_i to chi_e according to the assumptions of the
    # GyroBohm model.
    chi_i_gB = 0.5 * chi_e_gB

    # Calibrated transport coefficients
    chi_e_bohm = (
        transport_dynamic_runtime_params.chi_e_bohm_coeff
        * transport_dynamic_runtime_params.chi_e_bohm_multiplier
        * chi_e_B
    )
    chi_e_gyrobohm = (
        transport_dynamic_runtime_params.chi_e_gyrobohm_coeff
        * transport_dynamic_runtime_params.chi_e_gyrobohm_multiplier
        * chi_e_gB
    )

    chi_i_bohm = (
        transport_dynamic_runtime_params.chi_i_bohm_coeff
        * transport_dynamic_runtime_params.chi_i_bohm_multiplier
        * chi_i_B
    )
    chi_i_gyrobohm = (
        transport_dynamic_runtime_params.chi_i_gyrobohm_coeff
        * transport_dynamic_runtime_params.chi_i_gyrobohm_multiplier
        * chi_i_gB
    )

    # Total heat transport (combined contributions)
    chi_e = chi_e_gyrobohm + chi_e_bohm
    chi_i = chi_i_gyrobohm + chi_i_bohm

    # Electron diffusivity
    weighting = (
        transport_dynamic_runtime_params.D_face_c1
        + (
            transport_dynamic_runtime_params.D_face_c2
            - transport_dynamic_runtime_params.D_face_c1
        )
        * geo.rho_face_norm
    )

    # Diffusion: d_face_el is zero on-axis by definition.
    # We add a small epsilon to the denominator to avoid cases where
    # chi_i + chi_e = 0.
    d_face_el = jnp.concatenate([
        jnp.zeros(1),
        weighting[1:]
        * chi_e[1:]
        * chi_i[1:]
        / (chi_e[1:] + chi_i[1:] + constants_module.CONSTANTS.eps),
    ])

    # Electron convectivity set proportional to the electron diffusivity
    v_face_el = transport_dynamic_runtime_params.V_face_coeff * d_face_el

    return transport_model_lib.TurbulentTransport(
        chi_face_ion=chi_i,
        chi_face_el=chi_e,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
        chi_face_el_bohm=chi_e_bohm,
        chi_face_el_gyrobohm=chi_e_gyrobohm,
        chi_face_ion_bohm=chi_i_bohm,
        chi_face_ion_gyrobohm=chi_i_gyrobohm,
    )

  def __hash__(self):
    # All BohmGyroBohmModels are equivalent and can hash the same
    return hash('BohmGyroBohmModel')

  def __eq__(self, other):
    return isinstance(other, BohmGyroBohmTransportModel)
