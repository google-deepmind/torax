# Copyright 2026 DeepMind Technologies Limited
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

"""Centralized gyroBohm normalization conversions for the gyaradax plugin.

gyaradax follows GKW conventions and its outputs stay in GKW units; every
conversion to TORAX happens here, at runtime, on the TORAX side. Other unit
systems of interest (QuaLiKiz/QLKNN, for calibrating the gyaradax-QL head on
the QLKNN dataset) are expressed in the same framework so any gyroBohm flux
can be converted between conventions with one function call.

A gyroBohm convention is fixed by three choices:

1. **Thermal-velocity definition.** GKW uses ``v_th = sqrt(2 T / m)``;
   TORAX's quasilinear machinery and QuaLiKiz use ``sqrt(T / m)`` (via
   ``chiGB = sqrt(m) T^{3/2} / (q_e B)^2 / L_ref``). The gyroBohm heat-flux
   reference is ``Q_ref = n T v_th rho_*^2`` with ``rho_* = m v_th / (q_e B
   L_ref)``, i.e. ``Q_ref ∝ v_th^3``, so the sqrt(2) velocity choice changes
   the flux reference by ``(sqrt 2)^3 = 2 sqrt(2)``.
2. **Reference length** in ``rho_*`` and the flux-gradient relation. GKW uses
   the major radius ``R``; TORAX's quasilinear base accepts it as an argument
   (the plugin passes ``R_major`` for both the gradient and the flux
   reference); QuaLiKiz/QLKNN use ``a_minor`` for the flux reference while
   keeping ``R``-based logarithmic gradients. ``Q_ref ∝ 1 / L_ref^2``.
3. **chi-factor reference temperature** — the temperature inside the
   gyroBohm velocity/diffusivity factor, entering as ``T_chi^{3/2}``. GKW
   uses the reference-species temperature (= ``T_i`` here, GKW manual
   diagnostics: ``R_ref Q_s = n_s T_s rho_*^2 v_thref (I_2+J_2+K_2)`` with
   ``rho_*, v_thref`` built from ``T_ref``); the QLKNN-hyper dataset uses
   ``T_e`` (paper Eq. B1: ``c_GB = sqrt(A_i0 m_p) T_e^{1.5} / (q_e^2 B_0^2
   a)``).

In every convention the heat-flux reference additionally carries the
**species** temperature linearly (GKW: ``n_s T_s ...``; QLKNN Eq. B5:
``q_GB = a / (n_s T_s c_GB) q_SI``). For the channels exchanged here the
species is the main ion in all conventions, so the linear ``T_s`` factor
cancels in conversions and only the chi-factor temperature differs:

    factor = (2 sqrt 2)^{s_from - s_to} * (L_to / L_from)^2
             * (T_chi_from / T_chi_to)^{3/2}

with ``s = 1`` for sqrt(2)-velocity conventions. Normalized logarithmic
gradients are ``R/L_X`` (major-radius normalized, minor-radius derivative)
in GKW, QuaLiKiz, and the QLKNN dataset alike (paper Eq. B7 + Table I), so
they pass through unchanged; ``gradient_reference_length`` documents the
single choice used by the plugin.

Caveat on TORAX's own QLKNN wrapper: it builds chiGB with ``T_i`` ("due to
QLKNN training set normalization" refers to the a_minor choice), which
deviates from the paper's ``T_e^{3/2}`` by ``(T_e/T_i)^{3/2}`` when
``T_i != T_e``; QLKNN_HYPER below encodes the *paper* definition.

**Input parameters** (verified definition-by-definition):

| input | gyaradax/GKW            | TORAX plugin source                | QLKNN dataset      |
|-------|--------------------------|------------------------------------|--------------------|
| rlt   | R_ref/L_Ti, d/dr minor   | NormalizedLogarithmicGradients(rmid, R_major) | R/L_Ti (Eq. B7) |
| rln   | R_ref/L_n                | same, n_e                          | R/L_n              |
| q     | safety factor            | q_face                             | q                  |
| shat  | (r/q) dq/dr              | calc_s_rmid = -r_mid d(iota)/dr_mid / iota (identical) | smag (same) |
| eps   | r/R_ref                  | epsilon_face = (R_out-R_in)/(R_out+R_in) (= r_mid/R_local; differs from r/R_axis by the Shafranov shift only) | r/R (Table I) |
| beta  | 2 mu0 n_ref T_ref/B_ref^2| fixed 0 (ES) — revisit for the EM tier | not in dataset |

All identical up to the Shafranov-shift nuance in eps; no input conversion
is required between the three systems.

**Kinetic species and electromagnetic channels.** GKW's flux normalization
(manual diagnostics.tex) applies one prefactor ``n_s T_s rho_*^2 v_thref``
to the *sum* of the electrostatic (I), A_parallel flutter (J), and
B_parallel compressional (K) channels — so EM flux contributions convert
with exactly the same factor as electrostatic ones. Per species, the linear
``n_s T_s`` factor appears identically in TORAX's `_make_core_transport`
implied flux reference (chi_s from q_s / (R/L_Ts) keeps n_s T_s linear with
the chi-factor temperature unchanged), so the conversion factor below is
species-independent: the same 2*sqrt(2) applies to ion and electron heat
channels and to ES+EM totals when kinetic electrons arrive. (Empirical
anchor: gyaradax same-state ES/flutter/compressional per-species fluxes
match GKW's printed diagnostics to ~1e-6 at beta up to 0.01.)

References: GKW manual doc/manual/diagnostics.tex (flux normalization);
gyaradax docs/NOTES.md sec. 1.1; van de Plassche et al., PoP 27, 012305
(2020), Appendix B, for QLKNN-hyper.
"""

import dataclasses
import math
from typing import Literal

import chex
import jax.numpy as jnp

from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.transport_model.quasilinear_transport_model import calculate_chiGB
from torax._src.transport_model.quasilinear_transport_model import NormalizedLogarithmicGradients
from torax._src.transport_model.quasilinear_transport_model import QuasilinearInputs

# (sqrt 2)^3: GKW's v_th = sqrt(2T/m) versus sqrt(T/m) in the flux reference.
VTH_SQRT2_FLUX_FACTOR = 2.0 * math.sqrt(2.0)


@dataclasses.dataclass(frozen=True)
class GyroBohmConvention:
  """One gyroBohm unit system (see module docstring for the three choices)."""

  name: str
  flux_reference_length: Literal['R_major', 'a_minor']
  vth_includes_sqrt2: bool
  # T inside the gyroBohm chi factor (^3/2); linear species-T cancels
  chi_reference_temperature: Literal['T_i', 'T_e']


# gyaradax / GKW native units: everything the gyaradax-QL rule returns.
GYARADAX = GyroBohmConvention(
    name='gyaradax-gkw',
    flux_reference_length='R_major',
    vth_includes_sqrt2=True,
    chi_reference_temperature='T_i',
)

# what the plugin feeds TORAX's quasilinear base: R_major, sqrt(T/m), T_i
TORAX_QL = GyroBohmConvention(
    name='torax-quasilinear',
    flux_reference_length='R_major',
    vth_includes_sqrt2=False,
    chi_reference_temperature='T_i',
)

# QLKNN-hyper dataset units (van de Plassche 2020, Eqs. B1/B5: a_minor, T_e)
QLKNN_HYPER = GyroBohmConvention(
    name='qlknn-hyper',
    flux_reference_length='a_minor',
    vth_includes_sqrt2=False,
    chi_reference_temperature='T_e',
)


def flux_reference_length(
    convention: GyroBohmConvention, geo: geometry_lib.Geometry
) -> chex.Numeric:
  """The L_ref entering this convention's gyroBohm flux reference."""
  if convention.flux_reference_length == 'R_major':
    return geo.R_major
  return geo.a_minor


def gradient_reference_length(geo: geometry_lib.Geometry) -> chex.Numeric:
  """Reference length for normalized logarithmic gradients (R/L_T, R/L_n).

  GKW, QuaLiKiz, and the QLKNN dataset all use the major radius here; the
  plugin keeps that single convention end to end.
  """
  return geo.R_major


def chi_reference_temperature(
    convention: GyroBohmConvention, core_profiles: state.CoreProfiles
) -> chex.Numeric:
  """chi-factor T_ref on the face grid for this convention [keV]."""
  if convention.chi_reference_temperature == 'T_i':
    return core_profiles.T_i.face_value()
  return core_profiles.T_e.face_value()


def chi_gb(
    convention: GyroBohmConvention,
    core_profiles: state.CoreProfiles,
    geo: geometry_lib.Geometry,
) -> chex.Numeric:
  """gyroBohm diffusivity chiGB for this convention on the face grid.

  Note chiGB always uses the sqrt(T/m) form (TORAX's `calculate_chiGB`); the
  sqrt(2) velocity choice is a property of the *flux* reference and is
  applied by `gb_flux_conversion_factor`, never baked into chiGB.
  """
  return calculate_chiGB(
      reference_temperature=chi_reference_temperature(
          convention, core_profiles
      ),
      reference_magnetic_field=geo.B_0,
      reference_mass=core_profiles.A_i,
      reference_length=flux_reference_length(convention, geo),
  )


def gb_flux_conversion_factor(
    from_convention: GyroBohmConvention,
    to_convention: GyroBohmConvention,
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles | None = None,
) -> chex.Numeric:
  """Multiplicative factor converting a gyroBohm flux between conventions.

  factor = Q_ref(from) / Q_ref(to)
         = (2*sqrt(2))^(s_f - s_t) * (L_to/L_from)^2 * (T_chi_f/T_chi_t)^(3/2)

  (The linear species-temperature factor of the flux reference is T_i in all
  conventions handled here and cancels; see module docstring.)

  Args:
    from_convention: convention the flux value is currently in.
    to_convention: target convention.
    geo: TORAX geometry (for the reference lengths).
    core_profiles: required only when the two conventions use different
      reference temperatures (T_i vs T_e); the factor is then a face-grid
      profile rather than a scalar.

  Returns:
    Scalar (or face-grid array when T refs differ) conversion factor.
  """
  s_from = 1.0 if from_convention.vth_includes_sqrt2 else 0.0
  s_to = 1.0 if to_convention.vth_includes_sqrt2 else 0.0
  factor = VTH_SQRT2_FLUX_FACTOR ** (s_from - s_to)

  l_from = flux_reference_length(from_convention, geo)
  l_to = flux_reference_length(to_convention, geo)
  factor = factor * (l_to / l_from) ** 2

  if (
      from_convention.chi_reference_temperature
      != to_convention.chi_reference_temperature
  ):
    if core_profiles is None:
      raise ValueError(
          f'Converting {from_convention.name} -> {to_convention.name} changes '
          'the chi reference temperature; core_profiles is required.'
      )
    t_from = chi_reference_temperature(from_convention, core_profiles)
    t_to = chi_reference_temperature(to_convention, core_profiles)
    factor = factor * (t_from / t_to) ** 1.5
  return factor


def convert_gb_flux(
    value: chex.Numeric,
    from_convention: GyroBohmConvention,
    to_convention: GyroBohmConvention,
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles | None = None,
) -> chex.Numeric:
  """Converts a dimensionless gyroBohm flux between conventions.

  The (T_chi)^{3/2} term covers heat AND particle fluxes (it comes from
  c_GB; the linear species-T factor appears only in the heat-flux reference
  and is T_i in all conventions handled here, GKW manual / QLKNN Eq. B2 vs
  B5). Heat flux is the calibrated channel today.
  """
  return value * gb_flux_conversion_factor(
      from_convention, to_convention, geo, core_profiles
  )


def build_quasilinear_inputs(
    core_profiles: state.CoreProfiles, geo: geometry_lib.Geometry
) -> QuasilinearInputs:
  """TORAX QuasilinearInputs in the plugin's (TORAX_QL) convention.

  chiGB and the logarithmic gradients are both referenced to R_major so the
  base class's flux -> chi conversion is consistent with gyaradax's
  major-radius gradient convention after `convert_gb_flux(GYARADAX ->
  TORAX_QL)` has been applied to the fluxes.
  """
  log_grads = NormalizedLogarithmicGradients.from_profiles(
      core_profiles=core_profiles,
      radial_coordinate=geo.r_mid,
      radial_face_coordinate=geo.r_mid_face,
      reference_length=gradient_reference_length(geo),
  )
  return QuasilinearInputs(
      chiGB=chi_gb(TORAX_QL, core_profiles, geo),
      Rmaj=geo.R_major,
      Rmin=geo.a_minor,
      lref_over_lti=log_grads.lref_over_lti,
      lref_over_lte=log_grads.lref_over_lte,
      lref_over_lne=log_grads.lref_over_lne,
      lref_over_lni0=log_grads.lref_over_lni0,
      lref_over_lni1=log_grads.lref_over_lni1,
  )
