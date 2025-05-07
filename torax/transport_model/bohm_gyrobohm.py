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
import chex
from jax import numpy as jnp
from torax import array_typing
from torax import constants as constants_module
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  chi_e_bohm_coeff: array_typing.ScalarFloat
  chi_e_gyrobohm_coeff: array_typing.ScalarFloat
  chi_i_bohm_coeff: array_typing.ScalarFloat
  chi_i_gyrobohm_coeff: array_typing.ScalarFloat
  D_face_c1: array_typing.ScalarFloat
  D_face_c2: array_typing.ScalarFloat
  V_face_coeff: array_typing.ScalarFloat
  chi_e_bohm_multiplier: array_typing.ScalarFloat
  chi_e_gyrobohm_multiplier: array_typing.ScalarFloat
  chi_i_bohm_multiplier: array_typing.ScalarFloat
  chi_i_gyrobohm_multiplier: array_typing.ScalarFloat


class BohmGyroBohmTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport according to the Bohm + gyro-Bohm Model."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_outputs: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    r"""Calculates transport coefficients using the BohmGyroBohm model.

    We use the implementation from Tholerus et al, Section 3.3.
    https://doi.org/10.1088/1741-4326/ad6ea2

    A description is provided in physics_models.rst.
    https://torax.readthedocs.io/en/latest/physics_models.html

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_outputs: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    del pedestal_model_outputs  # Unused.
    # pylint: disable=invalid-name
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    true_n_e_face = (
        core_profiles.n_e.face_value()
        * dynamic_runtime_params_slice.numerics.density_reference
    )
    true_n_e_grad_face = (
        core_profiles.n_e.face_grad()
        * dynamic_runtime_params_slice.numerics.density_reference
    )

    # Bohm term of heat transport
    chi_e_B = (
        geo.r_mid_face
        * core_profiles.q_face**2
        / (constants_module.CONSTANTS.qe * geo.B_0 * true_n_e_face)
        * (
            jnp.abs(true_n_e_grad_face) * core_profiles.temp_el.face_value()
            + jnp.abs(core_profiles.temp_el.face_grad()) * true_n_e_face
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
            dynamic_runtime_params_slice.plasma_composition.main_ion.avg_A / 2
        )
        * jnp.sqrt(core_profiles.temp_el.face_value() * 1e3)
        / geo.B_0**2
        * jnp.abs(core_profiles.temp_el.face_grad() * 1e3)
        / geo.rho_b
    )

    # Set proportionality of chi_i to chi_e according to the assumptions of the
    # GyroBohm model.
    chi_i_gB = 0.5 * chi_e_gB

    # Calibrated transport coefficients
    chi_e_bohm = (
        dynamic_runtime_params_slice.transport.chi_e_bohm_coeff
        * dynamic_runtime_params_slice.transport.chi_e_bohm_multiplier
        * chi_e_B
    )
    chi_e_gyrobohm = (
        dynamic_runtime_params_slice.transport.chi_e_gyrobohm_coeff
        * dynamic_runtime_params_slice.transport.chi_e_gyrobohm_multiplier
        * chi_e_gB
    )

    chi_i_bohm = (
        dynamic_runtime_params_slice.transport.chi_i_bohm_coeff
        * dynamic_runtime_params_slice.transport.chi_i_bohm_multiplier
        * chi_i_B
    )
    chi_i_gyrobohm = (
        dynamic_runtime_params_slice.transport.chi_i_gyrobohm_coeff
        * dynamic_runtime_params_slice.transport.chi_i_gyrobohm_multiplier
        * chi_i_gB
    )

    # Total heat transport (combined contributions)
    chi_e = chi_e_gyrobohm + chi_e_bohm
    chi_i = chi_i_gyrobohm + chi_i_bohm

    # Electron diffusivity
    weighting = (
        dynamic_runtime_params_slice.transport.D_face_c1
        + (
            dynamic_runtime_params_slice.transport.D_face_c2
            - dynamic_runtime_params_slice.transport.D_face_c1
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
    v_face_el = dynamic_runtime_params_slice.transport.V_face_coeff * d_face_el

    return state.CoreTransport(
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
