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
from typing import Self
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
from torax._src.transport_model import component
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_coeffs


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
      two_point_mask: array_typing.BoolVectorFace | None = None,
  ) -> Self:
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
          two_point_mask=two_point_mask,
      )
    fi_grads = {}
    for fi in core_profiles.fast_ions:
      key = f"{fi.source}_{fi.species}"
      lref_over_ln = calculate_normalized_logarithmic_gradient(
          var=fi.n,
          radial_coordinate=radial_coordinate,
          radial_face_coordinate=radial_face_coordinate,
          reference_length=reference_length,
          two_point_mask=two_point_mask,
      )
      lref_over_lt = calculate_normalized_logarithmic_gradient(
          var=fi.T,
          radial_coordinate=radial_coordinate,
          radial_face_coordinate=radial_face_coordinate,
          reference_length=reference_length,
          two_point_mask=two_point_mask,
      )
      fi_grads[key] = {
          "lref_over_ln": lref_over_ln,
          "lref_over_lt": lref_over_lt,
      }
    gradients["fast_ion_gradients"] = fi_grads
    return cls(**gradients)


# pylint: disable=invalid-name
@jax.jit
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.ComponentRuntimeParams):
  """Shared parameters for Quasilinear models."""

  DV_effective: bool = dataclasses.field(metadata={"static": True})
  An_min: float
  DV_effective_smooth_width: float


@jax.jit
def calculate_normalized_logarithmic_gradient(
    var: cell_variable.CellVariable,
    radial_coordinate: jax.Array,
    radial_face_coordinate: jax.Array,
    reference_length: jax.Array,
    two_point_mask: array_typing.BoolVectorFace | None = None,
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
          two_point_mask=two_point_mask,
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


def calculate_dv_effective(
    particle_flux_SI: jax.Array,
    normalized_particle_flux: jax.Array,
    n_e: cell_variable.CellVariable,
    geo: geometry.Geometry,
    gradient_reference_length: array_typing.FloatScalar,
    An_min: array_typing.FloatScalar,
    DV_effective_smooth_width: array_typing.FloatScalar,
    two_point_mask: array_typing.BoolVectorFace | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Calculates effective diffusivity and convectivity for DV_effective mode.

  Splits the particle flux between pure effective diffusion (`D_eff`) and pure
  effective convection (`V_eff`):
    d_face_el = diffusion_weight * D_eff
    v_face_el = (1 - diffusion_weight) * V_eff
  where `diffusion_weight = grad_weight * flux_weight` in [0, 1] assigns
  transport to convection for small density gradients (|L_ref / L_ne| < An_min)
  or up-gradient flux, and to diffusion otherwise.

  When `DV_effective_smooth_width` == 0.0, `grad_weight` and `flux_weight` are
  step functions. When `DV_effective_smooth_width` > 0.0, they use polynomial
  splines chosen so that `d_face_el` and `v_face_el` have continuous first
  derivatives (C1) across both transitions:
    1. `grad_weight`: Quartic spline `x^3 * (4 - 3*x)` on
       `x = clip(|L_ref / L_ne| / An_min, 0, 1)`. Because `D_eff` scales as
       `1 / (L_ref / L_ne)`, the `x^3` factor makes `grad_weight * D_eff` scale
       as `x^2`, giving zero value and zero derivative at zero density gradient
       along with zero derivative at `An_min`.
    2. `flux_weight`: Quadratic spline `x * (2 - x)` on
       `x = clip(downgrad_flux / DV_effective_smooth_width, 0, 1)`. Because
       `D_eff` and `V_eff` are already linear in flux, a linear factor `x` near
       zero flux makes `flux_weight * D_eff` quadratic in flux, giving zero
       derivative across flux reversal along with zero derivative at
       `DV_effective_smooth_width`.

  Args:
    particle_flux_SI: Electron particle flux in SI units [m^-2 s^-1].
    normalized_particle_flux: Dimensionless GyroBohm-normalized particle flux.
    n_e: Electron density CellVariable [m^-3].
    geo: Torus geometry.
    gradient_reference_length: Reference length L_ref for An_min [m].
    An_min: Normalized logarithmic gradient threshold |L_ref / L_ne| below which
      transport transitions from diffusion to convection.
    DV_effective_smooth_width: Particle flux width in dimensionless
      GyroBohm-normalized units (Gamma_e / Gamma_GB) over which down-gradient
      transport transitions from convection to diffusion. If 0.0, uses a sharp
      step-function transition.
    two_point_mask: Optional boolean mask for 2-point face gradients.

  Returns:
    Tuple of (d_face_el [m^2/s], v_face_el [m/s]).
  """
  n_e_face = n_e.face_value()
  dn_e_drhon = n_e.face_grad(two_point_mask=two_point_mask)
  dn_e_drhon_safe = jnp.where(dn_e_drhon == 0.0, 1.0, dn_e_drhon)

  # Effective diffusivity (pure D) and convectivity (pure V) that each
  # individually reproduce the full particle flux.
  D_eff = jnp.where(
      dn_e_drhon == 0.0,
      0.0,
      -particle_flux_SI / (dn_e_drhon_safe * geo.g1_over_vpr2_face * geo.rho_b),
  )
  V_eff = particle_flux_SI / (n_e_face * geo.g0_over_vpr_face * geo.rho_b)

  dn_e_drmid = n_e.face_grad(
      x=geo.r_mid,
      x_left=geo.r_mid_face[0],
      x_right=geo.r_mid_face[-1],
      two_point_mask=two_point_mask,
  )
  lref_over_lne = -dn_e_drmid * gradient_reference_length / n_e_face

  # 1. Smooth density-gradient weight: 0 at zero gradient, 1 for
  # |lref_over_lne| >= An_min.
  norm_grad = jnp.clip(jnp.abs(lref_over_lne) / An_min, 0.0, 1.0)
  grad_weight = norm_grad**3 * (4.0 - 3.0 * norm_grad)

  # 2. Smooth down-gradient flux weight: 0 for up-gradient flux, 1 for
  # down-gradient flux >= DV_effective_smooth_width.
  sign_lref_over_lne = jnp.where(lref_over_lne >= 0.0, 1.0, -1.0)
  downgrad_flux = normalized_particle_flux * sign_lref_over_lne
  # Avoid division by zero if DV_effective_smooth_width == 0.0, where
  # sharp_diffusion_mask is used instead.
  smooth_width_safe = jnp.where(
      DV_effective_smooth_width > 0.0, DV_effective_smooth_width, 1.0
  )
  norm_flux = jnp.clip(downgrad_flux / smooth_width_safe, 0.0, 1.0)
  flux_weight = norm_flux * (2.0 - norm_flux)

  sharp_diffusion_mask = (jnp.abs(lref_over_lne) >= An_min) & (
      downgrad_flux >= 0.0
  )
  diffusion_weight = jnp.where(
      DV_effective_smooth_width == 0.0,
      sharp_diffusion_mask,
      grad_weight * flux_weight,
  )
  d_face_el = diffusion_weight * D_eff
  v_face_el = (1.0 - diffusion_weight) * V_eff
  return d_face_el, v_face_el


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
    # TODO(b/512128432): Extend the parameter space based on values ranges
    # observed following analysis of integrated modelling results.
    # TODO(b/512476967): Propagate and handle OOD warnings.
    species_factor, _ = fi_model.predict(inputs, clip_inputs=True)
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


class QuasilinearTransportModel(component.ComponentTransportModel):
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
      gradient_reference_length: array_typing.FloatScalar,
      gyrobohm_flux_reference_length: array_typing.FloatScalar,
      two_point_mask: array_typing.BoolVectorFace | None = None,
  ) -> transport_coeffs.TransportCoeffs:
    """Converts model output to TransportCoeffs."""
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

    if transport.DV_effective:
      d_face_el, v_face_el = calculate_dv_effective(
          particle_flux_SI=pfe_SI,
          normalized_particle_flux=pfe,
          n_e=core_profiles.n_e,
          geo=geo,
          gradient_reference_length=gradient_reference_length,
          An_min=transport.An_min,
          DV_effective_smooth_width=transport.DV_effective_smooth_width,
          two_point_mask=two_point_mask,
      )
    else:
      # Scaled D approach. Scale electron diffusivity to electron heat
      # conductivity (this has some physical motivations),
      # and set convection to then match total particle transport.
      # TODO(b/567403838): Create a helper function, calculate_d_scaled, and
      # fix gradient to use dn_e_drhon for consistency with TGLF in follow-up.
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

    return transport_coeffs.TransportCoeffs(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
