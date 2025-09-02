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
"""Base class for quasilinear models."""
import dataclasses
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib
import typing_extensions


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NormalizedLogarithmicGradients:
  """Normalized logarithmic gradients of plasma profiles.

  Defined as Lref/Lprofile. Lref is an arbitrary reference length [m].
  lprofile is each profile gradient length [m] defined as -1/grad(log(profile)),
  e.g. lti = -1/grad(log(ti)), i.e. lti = - ti / (dti/dr).
  The specific radial coordinate r used for the gradient is a user input.
  """

  lref_over_lti: array_typing.FloatVectorFace
  lref_over_lte: array_typing.FloatVectorFace
  lref_over_lne: array_typing.FloatVectorFace
  lref_over_lni0: array_typing.FloatVectorFace
  lref_over_lni1: array_typing.FloatVectorFace

  @classmethod
  def from_profiles(
      cls,
      core_profiles: state.CoreProfiles,
      radial_coordinate: jnp.ndarray,
      reference_length: jnp.ndarray,
  ) -> typing_extensions.Self:
    """Calculates the normalized logarithmic gradients."""
    gradients = {}
    for name, profile in {
        "lref_over_lti": core_profiles.T_i,
        "lref_over_lte": core_profiles.T_e,
        "lref_over_lne": core_profiles.n_e,
        "lref_over_lni0": core_profiles.n_i,
        "lref_over_lni1": core_profiles.n_impurity,
    }.items():
      gradients[name] = calculate_normalized_logarithmic_gradient(
          var=profile,
          radial_coordinate=radial_coordinate,
          reference_length=reference_length,
      )
    return cls(**gradients)


# pylint: disable=invalid-name
def calculate_chiGB(
    reference_temperature: array_typing.Array,
    reference_magnetic_field: chex.Numeric,
    reference_mass: chex.Numeric,
    reference_length: chex.Numeric,
) -> array_typing.Array:
  """Calculates the gyrobohm diffusivity.

  Different transport models make different choices for the reference
  temperature, magnetic field, and mass used for gyrobohm normalization.

  Args:
    reference_temperature: Reference temperature on the face grid [keV].
    reference_magnetic_field: Magnetic field strength [T].
    reference_mass: Reference ion mass [amu].
    reference_length: Reference length for normalization [m].

  Returns:
    Gyrobohm diffusivity as a array_typing.Array [dimensionless].
  """
  constants = constants_module.CONSTANTS
  return (
      (reference_mass * constants.mp) ** 0.5
      / (reference_magnetic_field * constants.qe) ** 2
      * (reference_temperature * constants.keV2J) ** 1.5
      / reference_length
  )


def calculate_alpha(
    core_profiles: state.CoreProfiles,
    q: array_typing.FloatVectorFace,
    reference_magnetic_field: chex.Numeric,
    normalized_logarithmic_gradients: NormalizedLogarithmicGradients,
) -> array_typing.FloatVectorFace:
  """Calculates the alpha_MHD parameter.

  alpha_MHD = Lref q^2 beta' , where beta' is the radial gradient of beta, the
  ratio of plasma pressure to magnetic pressure, Lref a reference length,
  and q is the safety factor. Lref is included within the
  NormalizedLogarithmicGradients.

  Args:
    core_profiles: CoreProfiles object containing plasma profiles.
    q: Safety factor.
    reference_magnetic_field: Magnetic field strength. Different transport
      models have different definitions of the specific magnetic field input.
    normalized_logarithmic_gradients: Normalized logarithmic gradients of plasma
      profiles.

  Returns:
    Alpha value as a array_typing.FloatVectorFace.
  """
  constants = constants_module.CONSTANTS

  factor_0 = (
      2 * constants.keV2J / reference_magnetic_field**2 * constants.mu0 * q**2
  )
  alpha = factor_0 * (
      core_profiles.T_e.face_value()
      * core_profiles.n_e.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lte
          + normalized_logarithmic_gradients.lref_over_lne
      )
      + core_profiles.n_i.face_value()
      * core_profiles.T_i.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lti
          + normalized_logarithmic_gradients.lref_over_lni0
      )
      + core_profiles.n_impurity.face_value()
      * core_profiles.T_i.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lti
          + normalized_logarithmic_gradients.lref_over_lni1
      )
  )
  return alpha


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Shared parameters for Quasilinear models."""

  DV_effective: bool
  An_min: float


def calculate_normalized_logarithmic_gradient(
    var: cell_variable.CellVariable,
    radial_coordinate: jax.Array,
    reference_length: jax.Array,
) -> jax.Array:
  """Calculates the normalized logarithmic gradient of a CellVariable on the face grid."""

  # var ~ 0 is only possible for ions (e.g. zero impurity density), and we
  # guard against possible division by zero.
  result = jnp.where(
      jnp.abs(var.face_value()) < constants_module.CONSTANTS.eps,
      constants_module.CONSTANTS.eps,
      -reference_length * var.face_grad(radial_coordinate) / var.face_value(),
  )

  # to avoid divisions by zero elsewhere in TORAX, if the gradient is zero
  result = jnp.where(
      jnp.abs(result) < constants_module.CONSTANTS.eps,
      constants_module.CONSTANTS.eps,
      result,
  )
  return result


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
  """Variables required to convert outputs to TORAX CoreTransport outputs."""

  chiGB: (
      array_typing.FloatVectorFace
  )  # gyrobohm diffusivity used for normalizations [m^2/s].
  Rmin: array_typing.FloatScalar  # minor radius [m].
  Rmaj: array_typing.FloatScalar  #  major radius [m].
  # Normalized logarithmic gradients of the plasma profiles.
  # See NormalizedLogarithmicGradients for details.
  lref_over_lti: array_typing.FloatVectorFace
  lref_over_lte: array_typing.FloatVectorFace
  lref_over_lne: array_typing.FloatVectorFace
  lref_over_lni0: array_typing.FloatVectorFace
  lref_over_lni1: array_typing.FloatVectorFace


class QuasilinearTransportModel(transport_model_lib.TransportModel):
  """Base class for quasilinear models."""

  def _make_core_transport(
      self,
      qi: jax.Array,
      qe: jax.Array,
      pfe: jax.Array,
      quasilinear_inputs: QuasilinearInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      gradient_reference_length: chex.Numeric,
      gyrobohm_flux_reference_length: chex.Numeric,
  ) -> transport_model_lib.TurbulentTransport:
    """Converts model output to TurbulentTransport."""
    constants = constants_module.CONSTANTS

    # conversion to SI units (note that n is normalized here)

    # Convert the electron particle flux from GB (pfe) to SI units.
    pfe_SI = (
        pfe
        * core_profiles.n_e.face_value()
        * quasilinear_inputs.chiGB
        / gyrobohm_flux_reference_length
    )

    # chi outputs in SI units.
    # chi[GB] = -Q[GB]/(Lref/LT), chi is heat diffusivity, Q is heat flux,
    # where Lref is the gyrobohm normalization length, LT the logarithmic
    # gradient length (unnormalized). For normalized_logarithmic_gradients, the
    # normalization length can in principle be different from the gyrobohm flux
    # reference length. e.g. in QuaLiKiz Ati = -Rmaj/LTi, but the
    # gyrobohm flux reference length in QuaLiKiz is Rmin.
    # In case they are indeed different we rescale the normalized logarithmic
    # gradient by the ratio of the two reference lengths.
    chi_face_ion = (
        ((gradient_reference_length / gyrobohm_flux_reference_length) * qi)
        / quasilinear_inputs.lref_over_lti
    ) * quasilinear_inputs.chiGB
    chi_face_el = (
        ((gradient_reference_length / gyrobohm_flux_reference_length) * qe)
        / quasilinear_inputs.lref_over_lte
    ) * quasilinear_inputs.chiGB

    # Effective D / Effective V approach.
    # For small density gradients or up-gradient transport, set pure effective
    # convection. Otherwise pure effective diffusion.
    def DV_effective_approach() -> tuple[jax.Array, jax.Array]:
      # The geo.rho_b is to unnormalize the face_grad.
      Deff = -pfe_SI / (
          core_profiles.n_e.face_grad() * geo.g1_over_vpr2_face * geo.rho_b
          + constants.eps
      )
      Veff = pfe_SI / (
          core_profiles.n_e.face_value() * geo.g0_over_vpr_face * geo.rho_b
      )
      Deff_mask = (
          ((pfe >= 0) & (quasilinear_inputs.lref_over_lne >= 0))
          | ((pfe < 0) & (quasilinear_inputs.lref_over_lne < 0))
      ) & (abs(quasilinear_inputs.lref_over_lne) >= transport.An_min)
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
          pfe_SI / core_profiles.n_e.face_value()
          - quasilinear_inputs.lref_over_lne
          * d_face_el
          / gradient_reference_length
          * geo.g1_over_vpr2_face
          * geo.rho_b**2
      ) / (geo.g0_over_vpr_face * geo.rho_b)
      return d_face_el, v_face_el

    d_face_el, v_face_el = jax.lax.cond(
        transport.DV_effective,
        DV_effective_approach,
        Dscaled_approach,
    )
    return transport_model_lib.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
