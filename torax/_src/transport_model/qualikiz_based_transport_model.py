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
"""Base class and utils for Qualikiz-based models."""
import dataclasses
import enum

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import jax_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import collisions
from torax._src.physics import psi_calculations
from torax._src.physics import rotation
from torax._src.transport_model import quasilinear_transport_model


class RotationMode(enum.StrEnum):
  """Defines how the rotation correction is applied.

  OFF: No rotation correction is applied.
  HALF_RADIUS: The rotation correction is only applied to the outer
    half of the radius (rhon > 0.5).
  FULL_RADIUS: The rotation correction is applied everywhere.
  """
  OFF = 'off'
  HALF_RADIUS = 'half_radius'
  FULL_RADIUS = 'full_radius'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  """Shared parameters for Qualikiz-based models."""

  collisionality_multiplier: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool
  rotation_multiplier: float
  rotation_mode: RotationMode = dataclasses.field(metadata={'static': True})
  output_mode_contributions: bool = dataclasses.field(
    default=False, metadata={'static': True}
  )

# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QualikizInputs(quasilinear_transport_model.QuasilinearInputs):
  """Inputs to Qualikiz-based models."""

  Z_eff_face: array_typing.FloatVectorFace
  q: array_typing.FloatVectorFace
  smag: array_typing.FloatVectorFace
  x: array_typing.FloatVectorFace
  Ti_Te: array_typing.FloatVectorFace
  log_nu_star_face: array_typing.FloatVectorFace
  normni: array_typing.FloatVectorFace
  alpha: array_typing.FloatVectorFace
  epsilon: array_typing.FloatVectorFace
  gamma_E_GB: array_typing.FloatVectorFace
  gamma_E_QLK: array_typing.FloatVectorFace

  # Also define the logarithmic gradients using standard QuaLiKiz notation.
  @property
  def Ati(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lti

  @property
  def Ate(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lte

  @property
  def Ane(self) -> array_typing.Array:
    return self.lref_over_lne

  @property
  def Ani0(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lni0

  @property
  def Ani1(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lni1


class QualikizBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for Qualikiz-based transport models."""

  def _prepare_qualikiz_inputs(
      self,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      poloidal_velocity_multiplier: array_typing.FloatScalar,
  ) -> QualikizInputs:
    """Prepare Qualikiz inputs."""
    constants = constants_module.CONSTANTS

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = geo.r_mid
    rmid_face = geo.r_mid_face

    # gyrobohm diffusivity
    # (defined here with Lref=a_minor due to QLKNN training set normalization)
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.T_i.face_value(),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.A_i,
        reference_length=geo.a_minor,
    )

    # transport coefficients from the qlknn-hyper-10D model
    # (K.L. van de Plassche PoP 2020)

    # set up input vectors (all as jax.numpy arrays on face grid)

    # Calculate normalized logarithmic gradients
    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=rmid,
        radial_face_coordinate=rmid_face,
        reference_length=geo.R_major,
    )

    q = core_profiles.q_face

    # Due to QuaLikiz geometry assumptions, we need to calculate s with respect
    # to the midplane average, and not use the standard s_face from CoreProfiles
    smag = psi_calculations.calc_s_rmid(
        geo,
        core_profiles.psi,
    )

    # Inverse aspect ratio.
    epsilon = geo.epsilon_face
    # Local normalized radius.
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)

    # Ion to electron temperature ratio
    Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()

    # logarithm of normalized collisionality
    nu_star = collisions.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        collisionality_multiplier=transport.collisionality_multiplier,
    )
    log_nu_star_face = jnp.log10(nu_star)

    # calculate alpha for magnetic shear correction (see S. van Mulders NF 2021)
    alpha = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        q=q,
        reference_magnetic_field=geo.B_0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )

    # to approximate impact of Shafranov shift. From van Mulders Nucl. Fusion
    # 2021.
    smag = jnp.where(
        transport.smag_alpha_correction,
        smag - alpha / 2,
        smag,
    )

    # very basic ad-hoc sawtooth model
    smag = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        0.1,
        smag,
    )

    q = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        1,
        q,
    )

    smag = jnp.where(
        jnp.logical_and(
            transport.avoid_big_negative_s,
            smag - alpha < -0.2,
        ),
        alpha - 0.2,
        smag,
    )
    normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value()

    def _get_v_ExB():
      if transport.rotation_mode == RotationMode.OFF:
        return jnp.zeros_like(core_profiles.q_face)
      v_ExB, _, _ = rotation.calculate_rotation(
          T_i=core_profiles.T_i,
          psi=core_profiles.psi,
          n_i=core_profiles.n_i,
          q_face=core_profiles.q_face,
          Z_eff_face=core_profiles.Z_eff_face,
          Z_i_face=core_profiles.Z_i_face,
          toroidal_velocity=core_profiles.toroidal_velocity,
          pressure_thermal_i=core_profiles.pressure_thermal_i,
          geo=geo,
          poloidal_velocity_multiplier=poloidal_velocity_multiplier,
      )
      v_ExB = transport.rotation_multiplier * v_ExB
      if transport.rotation_mode == RotationMode.HALF_RADIUS:
        # Only consider contribution from the outer half-radius (rho > 0.5).
        v_ExB = v_ExB * jnp.where(geo.rho_face_norm > 0.5, 1, 0)
      return v_ExB

    # gamma_E_SI = r / q * d(v_ExB * q / r)/dr
    v_ExB = _get_v_ExB()
    # Computing gradient on the cell grid for better numerical accuracy.
    value_face = v_ExB * q / rmid_face
    cv = cell_variable.CellVariable(
        value=geometry.face_to_cell(value_face),
        dr=geo.drho_norm,
        right_face_constraint=value_face[-1],
        right_face_grad_constraint=None,
        left_face_constraint=None,
        left_face_grad_constraint=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )
    gamma_E_SI = rmid_face / q * cv.face_grad(
        x=rmid, x_left=rmid_face[0], x_right=rmid_face[-1]
    )

    # We need different normalizations for QuaLiKiz and QLKNN models.
    c_ref = jnp.sqrt(constants.keV_to_J / constants.m_amu)
    gamma_E_QLK = gamma_E_SI * (geo.R_major / c_ref)
    c_sou = jnp.sqrt(
        core_profiles.T_e.face_value()
        * constants.keV_to_J
        / (core_profiles.A_i * constants.m_amu)
    )
    gamma_E_GB = gamma_E_SI * (geo.a_minor / c_sou)

    return QualikizInputs(
        Z_eff_face=core_profiles.Z_eff_face,
        lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
        lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
        lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
        lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
        lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
        q=q,
        smag=smag,
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=geo.R_major,
        Rmin=geo.a_minor,
        alpha=alpha,
        epsilon=epsilon,
        gamma_E_GB=gamma_E_GB,
        gamma_E_QLK=gamma_E_QLK,
    )
