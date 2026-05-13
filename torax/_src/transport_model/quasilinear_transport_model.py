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

from collections.abc import Mapping
import dataclasses
import functools
import chex
from fusion_surrogates.fast_ion_stabilization import fast_ion_model
from fusion_surrogates.fast_ion_stabilization.models import registry as fi_registry
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import math_utils
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
  fast_ion_gradients: Mapping[str, Mapping[str, array_typing.FloatVectorFace]]

  @classmethod
  def from_profiles(
      cls,
      core_profiles: state.CoreProfiles,
      radial_coordinate: jnp.ndarray,
      radial_face_coordinate: jnp.ndarray,
      reference_length: jnp.ndarray,
  ) -> typing_extensions.Self:
    """Calculates the normalized logarithmic gradients."""
    gradients = {}
    for name, profile in {
        "lref_over_lti": core_profiles.T_i,
        "lref_over_lte": core_profiles.T_e,
        "lref_over_lne": core_profiles.n_e,
        "lref_over_lni0": core_profiles.n_i,
        "lref_over_lni1": core_profiles.n_impurity_thermal,
    }.items():
      gradients[name] = calculate_normalized_logarithmic_gradient(
          var=profile,
          radial_coordinate=radial_coordinate,
          radial_face_coordinate=radial_face_coordinate,
          reference_length=reference_length,
      )
    fi_grads = {}
    for fi in core_profiles.fast_ions:
      key = f"{fi.source}_{fi.species}"
      lref_over_ln = calculate_normalized_logarithmic_gradient(
          var=fi.n,
          radial_coordinate=radial_coordinate,
          radial_face_coordinate=radial_face_coordinate,
          reference_length=reference_length,
      )
      lref_over_lt = calculate_normalized_logarithmic_gradient(
          var=fi.T,
          radial_coordinate=radial_coordinate,
          radial_face_coordinate=radial_face_coordinate,
          reference_length=reference_length,
      )
      fi_grads[key] = {
          "lref_over_ln": lref_over_ln,
          "lref_over_lt": lref_over_lt,
      }
    gradients["fast_ion_gradients"] = fi_grads
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
      (reference_mass * constants.m_amu) ** 0.5
      / (reference_magnetic_field * constants.q_e) ** 2
      * (reference_temperature * constants.keV_to_J) ** 1.5
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
      2
      * constants.keV_to_J
      / reference_magnetic_field**2
      * constants.mu_0
      * q**2
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
      + core_profiles.n_impurity_thermal.face_value()
      * core_profiles.T_i.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lti
          + normalized_logarithmic_gradients.lref_over_lni1
      )
  )
  for fi in core_profiles.fast_ions:
    key = f"{fi.source}_{fi.species}"
    lref_over_ln = normalized_logarithmic_gradients.fast_ion_gradients[key][
        "lref_over_ln"
    ]
    lref_over_lT = normalized_logarithmic_gradients.fast_ion_gradients[key][
        "lref_over_lt"
    ]
    alpha += factor_0 * (
        fi.n.face_value() * fi.T.face_value() * (lref_over_ln + lref_over_lT)
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
    radial_face_coordinate: jax.Array,
    reference_length: jax.Array,
) -> jax.Array:
  """Face-grid normalized logarithmic gradient of a CellVariable."""

  # var ~ 0 is only possible for ions (e.g. zero impurity density), and we
  # guard against possible division by zero.
  result = jnp.where(
      jnp.abs(var.face_value()) < constants_module.CONSTANTS.eps,
      constants_module.CONSTANTS.eps,
      -reference_length
      * var.face_grad(
          x=radial_coordinate,
          x_left=radial_face_coordinate[0],
          x_right=radial_face_coordinate[-1],
      )
      / var.face_value(),
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


@functools.lru_cache(maxsize=2)
def _get_default_fi_stabilization_model(species: str):
  """Loads the default fast ion stabilization model for a species.

  Maps hydrogenic ions (H, D, T) to 'H' and helium isotopes (He3, He4)
  to 'He3' before looking up the default model.

  maxsize is the total number of fast ion models supported. maxsize will need
  to be increased if more distinct models are added.

  Args:
    species: Ion species name (e.g. 'He3', 'H', 'D').

  Returns:
    A loaded ``FastIonStabilizationModel`` instance.

  Raises:
    ValueError: If no default model exists for the mapped species.
  """
  if species in constants_module.HYDROGENIC_IONS:
    model_species = "H"
  elif species in ("He3", "He4"):
    model_species = "He3"
  else:
    model_species = species
  return (
      fast_ion_model.FastIonStabilizationModel.load_default_model_for_species(
          model_species
      )
  )


@functools.lru_cache(maxsize=2)
def _load_fi_stabilization_model(model: str):
  """Loads a fast ion stabilization model by name or path.

  Checks the model registry first; if not found, treats ``model``
  as a file path.

  maxsize is the total number of fast ion models supported. maxsize will need
  to be increased if more distinct models are added.

  Args:
    model: Registered model name or file path.

  Returns:
    A loaded ``FastIonStabilizationModel`` instance.
  """
  if model in fi_registry.MODELS:
    return fast_ion_model.FastIonStabilizationModel.load_model_from_name(model)
  return fast_ion_model.FastIonStabilizationModel.load_model_from_path(model)


def _compute_fast_ion_stabilization_factor(
    core_profiles: state.CoreProfiles,
    smag: jax.Array,
    q: jax.Array,
    normalized_logarithmic_gradients: NormalizedLogarithmicGradients,
    model_map: dict[str, str] | None = None,
) -> jax.Array:
  """Computes the combined fast ion stabilization factor for R/LTi.

  For each fast ion species, constructs model inputs (smag, q, n_fi/n_e,
  T_fi/T_e, R/L_{T_fi}) and predicts the ITG threshold modification factor.
  Returns the product over all species.

  Args:
    core_profiles: Core plasma profiles containing fast ion data.
    smag: Magnetic shear on the face grid.
    q: Safety factor on the face grid.
    normalized_logarithmic_gradients: Normalized logarithmic gradients
      containing fast ion gradient data.
    model_map: Mapping from species name to model name/path. If a species is not
      in the map, the default model for that species is loaded.

  Returns:
    Stabilization factor on the face grid.
  """
  if model_map is None:
    model_map = {}
  factor = jnp.ones_like(smag)
  for fast_ion in core_profiles.fast_ions:
    key = f"{fast_ion.source}_{fast_ion.species}"
    n_fi_over_ne = fast_ion.n.face_value() / core_profiles.n_e.face_value()
    t_fi_over_te = fast_ion.T.face_value() / core_profiles.T_e.face_value()
    lref_over_lt_fi = normalized_logarithmic_gradients.fast_ion_gradients[key][
        "lref_over_lt"
    ]
    # Feature ordering must match INPUT_FEATURES in fast_ion_model.py:
    # https://github.com/google-deepmind/fusion_surrogates/blob/main/fusion_surrogates/fast_ion_stabilization/fast_ion_model.py  # pylint: disable=line-too-long
    inputs = jnp.stack(
        [smag, q, n_fi_over_ne, t_fi_over_te, lref_over_lt_fi], axis=-1
    )
    species_model = model_map.get(fast_ion.species, "")
    if species_model:
      fi_model = _load_fi_stabilization_model(species_model)
    else:
      fi_model = _get_default_fi_stabilization_model(fast_ion.species)
    prediction = fi_model.predict(inputs)
    if isinstance(prediction, tuple):
      species_factor, _ = prediction
    else:
      species_factor = prediction
    species_factor = species_factor.squeeze(axis=-1)
    factor = factor * species_factor
  return factor


def apply_fast_ion_stabilization(
    core_profiles: state.CoreProfiles,
    smag: jax.Array,
    q: jax.Array,
    normalized_logarithmic_gradients: NormalizedLogarithmicGradients,
    transport: RuntimeParams,
) -> jax.Array:
  """Applies fast ion stabilization to the ion temperature gradient.

  The stabilization model returns a factor = 1 + n_fi/n_e * correction.
  The multiplier scales only the correction part:
    adjusted_factor = (factor - 1) * multiplier + 1

  Args:
    core_profiles: Core plasma profiles containing fast ion data.
    smag: Magnetic shear on the face grid.
    q: Safety factor on the face grid.
    normalized_logarithmic_gradients: Normalized logarithmic gradients.
    transport: Transport runtime parameters.

  Returns:
    Modified lref_over_lti with stabilization applied.
  """
  lref_over_lti = normalized_logarithmic_gradients.lref_over_lti
  model_map = dict(transport.fast_ion_stabilization_model)
  fi_stab_factor = _compute_fast_ion_stabilization_factor(
      core_profiles=core_profiles,
      smag=smag,
      q=q,
      normalized_logarithmic_gradients=normalized_logarithmic_gradients,
      model_map=model_map,
  )
  fi_stab_factor = (
      fi_stab_factor - 1
  ) * transport.fast_ion_stabilization_multiplier + 1
  return jnp.where(
      transport.fast_ion_stabilization,
      math_utils.safe_divide(num=lref_over_lti, denom=fi_stab_factor, eps=1e-7),
      lref_over_lti,
  )


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
