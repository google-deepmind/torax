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
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import collisions
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.physics import rotation
from torax._src.transport_model import quasilinear_transport_model
from torax._src.transport_model import transport_coeffs


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QualikizTransportModelOutput(transport_coeffs.TransportCoeffs):
  """QuaLiKiz transport coefficients with required sub-mode decompositions.

  Attributes:
    chi_face_ion_itg: ITG contribution for ion heat conductivity [m^2/s].
    chi_face_ion_tem: TEM contribution for ion heat conductivity [m^2/s].
    chi_face_el_itg: ITG contribution for electron heat conductivity [m^2/s].
    chi_face_el_tem: TEM contribution for electron heat conductivity [m^2/s].
    chi_face_el_etg: ETG contribution for electron heat conductivity [m^2/s].
    d_face_el_itg: ITG contribution for electron diffusivity [m^2/s].
    d_face_el_tem: TEM contribution for electron diffusivity [m^2/s].
    v_face_el_itg: ITG contribution for electron convection [m/s].
    v_face_el_tem: TEM contribution for electron convection [m/s].
  """

  chi_face_ion_itg: array_typing.FloatVectorFace
  chi_face_ion_tem: array_typing.FloatVectorFace
  chi_face_el_itg: array_typing.FloatVectorFace
  chi_face_el_tem: array_typing.FloatVectorFace
  chi_face_el_etg: array_typing.FloatVectorFace
  d_face_el_itg: array_typing.FloatVectorFace
  d_face_el_tem: array_typing.FloatVectorFace
  v_face_el_itg: array_typing.FloatVectorFace
  v_face_el_tem: array_typing.FloatVectorFace

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Converts QuaLiKiz decomposition channels to an OutputVar mapping."""
    out_dict = super().to_output_dict(context)
    out_dict[output_keys.CHI_ITG_I] = context.pack(
        output_keys.CHI_ITG_I, self.chi_face_ion_itg
    )
    out_dict[output_keys.CHI_TEM_I] = context.pack(
        output_keys.CHI_TEM_I, self.chi_face_ion_tem
    )
    out_dict[output_keys.CHI_ITG_E] = context.pack(
        output_keys.CHI_ITG_E, self.chi_face_el_itg
    )
    out_dict[output_keys.CHI_TEM_E] = context.pack(
        output_keys.CHI_TEM_E, self.chi_face_el_tem
    )
    out_dict[output_keys.CHI_ETG_E] = context.pack(
        output_keys.CHI_ETG_E, self.chi_face_el_etg
    )
    out_dict[output_keys.D_ITG_E] = context.pack(
        output_keys.D_ITG_E, self.d_face_el_itg
    )
    out_dict[output_keys.D_TEM_E] = context.pack(
        output_keys.D_TEM_E, self.d_face_el_tem
    )
    out_dict[output_keys.V_ITG_E] = context.pack(
        output_keys.V_ITG_E, self.v_face_el_itg
    )
    out_dict[output_keys.V_TEM_E] = context.pack(
        output_keys.V_TEM_E, self.v_face_el_tem
    )
    return out_dict


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
  max_normalized_collisionality: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool
  rotation_multiplier: float
  rotation_mode: RotationMode = dataclasses.field(metadata={'static': True})


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
  gamma_E_GB_poloidal_and_pressure: array_typing.FloatVectorFace
  gamma_E_GB_toroidal: array_typing.FloatVectorFace
  gamma_E_QLK: array_typing.FloatVectorFace
  mach_toroidal: array_typing.FloatVectorFace

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
      two_point_mask: array_typing.BoolVectorFace | None = None,
  ) -> QualikizInputs:
    """Constructs a `QualikizInputs` object from the TORAX state.

    Uses the midplane-averaged minor radius `r_mid` as the radial coordinate,
    with `L_ref = a_minor` for `chiGB` and `L_ref = R_major` for logarithmic
    gradients. Applies optional heuristic adjustments to avoid unreliable
    transport predictions and approximate missing physics: capping `nu_star`,
    clamping `q >= 1` and `smag = 0.1` where `q < 1` as a sawtooth proxy
    (`q_sawtooth_proxy`), and adjusting `smag` for the Shafranov shift
    (`smag - alpha / 2`) and strongly negative effective shear
    (`smag - alpha >= -0.2`) following S. van Mulders et al., Nucl. Fusion 61,
    086019 (2021).

    Args:
      transport: Runtime parameters for the QuaLiKiz-based transport model.
      geo: Torus geometry.
      core_profiles: Core plasma profiles.
      two_point_mask: Optional boolean face mask indicating where face gradients
        are calculated with 2-point central differences instead of 3-point.

    Returns:
      A `QualikizInputs` dataclass on the face grid.
    """
    constants = constants_module.CONSTANTS

    rmid = geo.r_mid
    rmid_face = geo.r_mid_face

    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.T_i.face_value(),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.A_i,
        reference_length=geo.a_minor,
    )

    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=rmid,
        radial_face_coordinate=rmid_face,
        reference_length=geo.R_major,
        two_point_mask=two_point_mask,
    )

    q = core_profiles.q_face
    smag = psi_calculations.calc_s_rmid(
        geo,
        core_profiles.psi,
    )

    epsilon = geo.epsilon_face
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)

    Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()

    nu_star = collisions.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        collisionality_multiplier=transport.collisionality_multiplier,
    )
    nu_star = jnp.minimum(nu_star, transport.max_normalized_collisionality)
    log_nu_star_face = jnp.log10(nu_star)

    alpha = formulas.calculate_alpha_mhd(
        core_profiles=core_profiles,
        geo=geo,
        two_point_mask=two_point_mask,
    )

    smag = jnp.where(
        transport.smag_alpha_correction,
        smag - alpha / 2,
        smag,
    )

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

    lref_over_lti = quasilinear_transport_model.apply_fast_ion_stabilization(
        core_profiles=core_profiles,
        smag=smag,
        q=q,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
        transport=transport,
    )

    if transport.rotation_mode == RotationMode.OFF:
      v_ExB = jnp.zeros_like(core_profiles.q_face)
      v_ExB_poloidal_and_pressure = jnp.zeros_like(core_profiles.q_face)
      v_ExB_toroidal = jnp.zeros_like(core_profiles.q_face)
    else:
      rotation_output = rotation.calculate_rotation(
          psi=core_profiles.psi,
          n_i=core_profiles.n_i,
          Z_i_face=core_profiles.Z_i_face,
          toroidal_angular_velocity=core_profiles.toroidal_angular_velocity,
          poloidal_velocity=core_profiles.poloidal_velocity,
          pressure_total_i=core_profiles.pressure_total_i,
          geo=geo,
      )
      v_ExB = rotation_output.v_ExB
      v_ExB_poloidal_and_pressure = rotation_output.v_ExB_poloidal_and_pressure
      v_ExB_toroidal = rotation_output.v_ExB_toroidal

    def _calc_gamma_E_SI(v_ExB_component):
      value_face = v_ExB_component * q / (rmid_face + constants.eps)
      cv = cell_variable.CellVariable(
          value=geometry.face_to_cell(value_face),
          face_centers=geo.rho_face_norm,
          right_face_constraint=value_face[-1],
          right_face_grad_constraint=None,
          left_face_constraint=None,
          left_face_grad_constraint=jnp.array(0.0, dtype=jax_utils.get_dtype()),
      )
      gamma_E_SI = (
          rmid_face
          / q
          * cv.face_grad(x=rmid, x_left=rmid_face[0], x_right=rmid_face[-1])
      )
      axis_ramp = jnp.minimum(
          (geo.rho_face_norm / 0.1) ** 2,
          1.0,
      )
      gamma_E_SI = gamma_E_SI * axis_ramp
      return gamma_E_SI * transport.rotation_multiplier

    gamma_E_SI = _calc_gamma_E_SI(v_ExB)
    gamma_E_SI_poloidal_and_pressure = _calc_gamma_E_SI(
        v_ExB_poloidal_and_pressure
    )
    gamma_E_SI_toroidal = _calc_gamma_E_SI(v_ExB_toroidal)

    c_ref = jnp.sqrt(constants.keV_to_J / constants.m_amu)
    gamma_E_QLK = gamma_E_SI * (geo.R_major / c_ref)
    mach_toroidal = (
        core_profiles.toroidal_angular_velocity.face_value()
        * geo.R_major_profile_face
        / c_ref
    )
    mach_toroidal = mach_toroidal * transport.rotation_multiplier

    c_sou = jnp.sqrt(
        core_profiles.T_e.face_value()
        * constants.keV_to_J
        / (core_profiles.A_i * constants.m_amu)
    )
    gamma_E_GB = gamma_E_SI * (geo.a_minor / c_sou)
    gamma_E_GB_poloidal_and_pressure = gamma_E_SI_poloidal_and_pressure * (
        geo.a_minor / c_sou
    )
    gamma_E_GB_toroidal = gamma_E_SI_toroidal * (geo.a_minor / c_sou)

    return QualikizInputs(
        Z_eff_face=core_profiles.Z_eff_face,
        lref_over_lti=lref_over_lti,
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
        gamma_E_GB_poloidal_and_pressure=gamma_E_GB_poloidal_and_pressure,
        gamma_E_GB_toroidal=gamma_E_GB_toroidal,
        gamma_E_QLK=gamma_E_QLK,
        mach_toroidal=mach_toroidal,
    )
