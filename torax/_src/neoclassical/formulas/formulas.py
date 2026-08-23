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
"""Common formulas used in neoclassical models."""

from collections.abc import Sequence
import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import collisions


# pylint: disable=invalid-name

FTrapModel = Literal['sauter', 'simple', 'RABBIT', 'LinLiu', 'numerical']

_NUMERICAL_F_TRAP_N_LAMBDA = 64


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IonSpeciesProfiles:
  """Density and temperature for one thermal ion or impurity species.

  Bootstrap drives use this species' ``n`` and ``T`` (and their logarithmic
  gradients) independently of other ions. When built from ``CoreProfiles``,
  TORAX currently assigns the evolved ``T_i`` to every thermal species.
  """

  n: cell_variable.CellVariable
  T: cell_variable.CellVariable


def make_thermal_pressure(
    n: cell_variable.CellVariable,
    T: cell_variable.CellVariable,
) -> cell_variable.CellVariable:
  """Builds a thermal pressure CellVariable from density and temperature."""
  kwargs = {}
  if (
      n.right_face_constraint is not None
      and T.right_face_constraint is not None
  ):
    kwargs['right_face_constraint'] = (
        n.right_face_constraint
        * T.right_face_constraint
        * constants.CONSTANTS.keV_to_J
    )
    kwargs['right_face_grad_constraint'] = None
  return cell_variable.CellVariable(
      value=n.value * T.value * constants.CONSTANTS.keV_to_J,
      face_centers=n.face_centers,
      **kwargs,
  )


def make_fractional_density(
    n_total: cell_variable.CellVariable,
    fraction: array_typing.FloatScalar | array_typing.FloatVectorCell,
    scaling: array_typing.FloatScalar | array_typing.FloatVectorCell = 1.0,
    scaling_face_edge: array_typing.FloatScalar | None = None,
) -> cell_variable.CellVariable:
  """Builds species density ``fraction * n_total * scaling``."""
  frac = jnp.asarray(fraction)
  scale = jnp.asarray(scaling)
  if frac.ndim == 0:
    frac_cell = frac
    frac_edge = frac
  else:
    frac_cell = frac
    frac_edge = frac[..., -1]
  if scale.ndim == 0:
    scale_cell = scale
    scale_edge = scale if scaling_face_edge is None else scaling_face_edge
  else:
    scale_cell = scale
    scale_edge = (
        scale[..., -1] if scaling_face_edge is None else scaling_face_edge
    )

  kwargs = {}
  if n_total.right_face_constraint is not None:
    kwargs['right_face_constraint'] = (
        frac_edge * n_total.right_face_constraint * scale_edge
    )
    kwargs['right_face_grad_constraint'] = None
  return cell_variable.CellVariable(
      value=frac_cell * n_total.value * scale_cell,
      face_centers=n_total.face_centers,
      **kwargs,
  )


def build_ion_species_from_core_profiles(
    core_profiles: state.CoreProfiles,
    *,
    subtract_fast_ions: bool = True,
) -> tuple[IonSpeciesProfiles, ...]:
  """Adapts CoreProfiles into per-species ``(n, T)`` for neoclassical models.

  Matches NEO's multi-species ion loop: each main ion and impurity is a
  separate contributor. TORAX evolves a single ``T_i``, so that profile is
  assigned to every species here; the bootstrap assembly still uses each
  species' own ``T`` and ``∇ln T``.

  Each main ion uses ``fraction * n_i``. Each impurity uses true density
  ``fraction * n_impurity * impurity_density_scaling``. When
  ``subtract_fast_ions`` is True, impurity densities exclude that species'
  fast-ion density when present.
  """
  ion_species: list[IonSpeciesProfiles] = []
  for fraction in core_profiles.main_ion_fractions.values():
    ion_species.append(
        IonSpeciesProfiles(
            n=make_fractional_density(core_profiles.n_i, fraction),
            T=core_profiles.T_i,
        )
    )

  scaling = core_profiles.impurity_density_scaling
  scaling_face_edge = core_profiles.impurity_density_scaling_face[..., -1]
  for symbol, fraction in core_profiles.impurity_fractions.items():
    n_s = make_fractional_density(
        core_profiles.n_impurity,
        fraction,
        scaling=scaling,
        scaling_face_edge=scaling_face_edge,
    )
    if subtract_fast_ions:
      for fast_ion in core_profiles.fast_ions:
        if fast_ion.species != symbol:
          continue
        n_val = n_s.value - fast_ion.n.value
        n_right = n_s.right_face_constraint
        if (
            n_right is not None
            and fast_ion.n.right_face_constraint is not None
        ):
          n_right = n_right - fast_ion.n.right_face_constraint
        n_s = cell_variable.CellVariable(
            value=n_val,
            face_centers=n_s.face_centers,
            right_face_constraint=n_right,
            right_face_grad_constraint=None,
        )
    ion_species.append(
        IonSpeciesProfiles(
            n=n_s,
            T=core_profiles.T_i,
        )
    )
  return tuple(ion_species)


def calculate_ion_density_sum_face(
    ion_species: Sequence[IonSpeciesProfiles],
    *,
    placeholder: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """NEO ``dens_sum``: sum of thermal ion/impurity densities on the face grid.

  Args:
    ion_species: Per-species thermal ion/impurity profiles. May be empty.
    placeholder: Face-shaped array used as the zeros template when summing
      (and as the return value shape when ``ion_species`` is empty).
  """
  dens_sum = jnp.zeros_like(placeholder)
  for species in ion_species:
    dens_sum = dens_sum + species.n.face_value()
  return dens_sum


def calculate_ion_pressure_sum_face(
    ion_species: Sequence[IonSpeciesProfiles],
    *,
    placeholder: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """NEO ``press_sum``: sum of thermal ion/impurity pressures on the face grid.

  Args:
    ion_species: Per-species thermal ion/impurity profiles. May be empty.
    placeholder: Face-shaped array used as the zeros template when summing
      (and as the return value shape when ``ion_species`` is empty).
  """
  press_sum = jnp.zeros_like(placeholder)
  for species in ion_species:
    press_sum = press_sum + make_thermal_pressure(
        species.n, species.T
    ).face_value()
  return press_sum


def calculate_Z_eff_from_ion_species(
    core_profiles: state.CoreProfiles,
    ion_species: Sequence[IonSpeciesProfiles],
) -> array_typing.FloatVectorFace:
  """Face ``Z_eff = Σ_s n_s Z_s² / n_e`` from per-species densities.

  Species order must match ``build_ion_species_from_core_profiles`` (main ions
  then impurities). Main-ion ``Z_s`` from ``ION_PROPERTIES_DICT``; impurity
  ``Z_s`` from ``charge_state_info_face.Z_per_species``.
  """
  n_e_face = core_profiles.n_e.face_value()
  Z_eff_num = jnp.zeros_like(n_e_face)
  species_idx = 0
  for symbol in core_profiles.main_ion_fractions:
    Z_s = constants.ION_PROPERTIES_DICT[symbol].Z
    Z_eff_num += ion_species[species_idx].n.face_value() * Z_s**2
    species_idx += 1
  for symbol in core_profiles.impurity_fractions:
    Z_s = core_profiles.charge_state_info_face.Z_per_species[symbol]
    Z_eff_num += ion_species[species_idx].n.face_value() * Z_s**2
    species_idx += 1
  if species_idx != len(ion_species):
    raise ValueError(
        'ion_species length does not match main-ion + impurity symbols in'
        f' core_profiles ({species_idx} symbols vs {len(ion_species)}'
        ' IonSpeciesProfiles).'
    )
  return Z_eff_num / n_e_face


def trapped_fraction_from_B(
    B: jax.Array,
    weights: jax.Array,
    n_lambda: int = _NUMERICAL_F_TRAP_N_LAMBDA,
) -> jax.Array:
  r"""Effective trapped fraction from the Sauter/Lin-Liu :math:`\lambda` integral.

  Evaluates
  :math:`f_t = 1 - \tfrac{3}{4}\langle B^2\rangle
  \int_0^{1/B_\mathrm{max}} \lambda\,d\lambda /
  \langle\sqrt{1 - \lambda B}\rangle`
  (O. Sauter et al., Phys. Plasmas 6, 2834 (1999), Eq. (12)).

  The flux-surface average uses the supplied quadrature weights (NEO
  Jacobian :math:`\\sqrt{g}\\,d\\theta`, equivalent to :math:`dl/B_p`).

  Args:
    B: Magnetic field strength samples on each flux surface. Shape
      ``(..., n_theta)``.
    weights: Non-negative FSA weights matching ``B``. Shape ``(..., n_theta)``.
    n_lambda: Number of midpoint quadrature points for the :math:`\lambda`
      integral.

  Returns:
    Trapped fraction with shape ``B.shape[:-1]``, clipped to ``[0, 1]``.
  """
  weight_sum = jnp.sum(weights, axis=-1)
  B_max = jnp.max(B, axis=-1)
  B_min = jnp.min(B, axis=-1)
  # Magnetic axis (and any surface with negligible |B| variation) has f_t = 0.
  is_uniform = B_max <= B_min * (1.0 + 1e-8)

  B2_avg = jnp.sum(B**2 * weights, axis=-1) / weight_sum

  # λ = λ_c x^2 with x∈(0,1) softens the bounce-point singularity at λ→λ_c.
  x = (jnp.arange(n_lambda) + 0.5) / n_lambda
  dx = 1.0 / n_lambda
  B_norm = B / B_max[..., jnp.newaxis]
  # Shape: (..., n_lambda, n_theta)
  under_sqrt = 1.0 - (x[..., jnp.newaxis] ** 2) * B_norm[..., jnp.newaxis, :]
  under_sqrt = jnp.maximum(under_sqrt, 0.0)
  sqrt_term = jnp.sqrt(under_sqrt)
  w = weights[..., jnp.newaxis, :]
  denom = jnp.sum(sqrt_term * w, axis=-1) / weight_sum[..., jnp.newaxis]
  denom = jnp.maximum(denom, 1e-30)

  lam_c = 1.0 / B_max
  # ∫ λ dλ / <√> = 2 λ_c² ∫_0^1 x³ / <√(1 - x² B/B_max)> dx
  integrand = 2.0 * (lam_c[..., jnp.newaxis] ** 2) * (x**3) / denom
  integral = jnp.sum(integrand, axis=-1) * dx

  f_t = 1.0 - 0.75 * B2_avg * integral
  f_t = jnp.clip(f_t, 0.0, 1.0)
  return jnp.where(is_uniform, 0.0, f_t)


def _neo_miller_B_and_weights(
    geo: geometry_lib.Geometry,
    q_face: array_typing.FloatVectorFace,
    n_theta: int = geometry_lib.N_THETA_SURFACE,
) -> tuple[jax.Array, jax.Array]:
  """NEO/GACODE-style Miller |B| and FSA weights on each flux surface.

  Builds an up-down-symmetric Miller surface from midplane ``R_in``,
  ``R_out``, elongation, and triangularity, including shaping shears.
  Then:

  * :math:`B_\\phi = F/R`
  * :math:`B_p = |d\\psi/dr|\\,|\\nabla r|/(2\\pi R)` with
    :math:`d\\psi/d\\rho_N = 2\\Phi_b\\rho_N/q` (TORAX ``calc_q_face``)
  * :math:`|B| = \\sqrt{B_\\phi^2+B_p^2}`
  * FSA weights :math:`\\sqrt{g}\\,d\\theta \\propto R|J|` (same as
    :math:`dl/B_p`)

  Args:
    geo: Magnetic geometry (face-grid shape quantities).
    q_face: Safety factor on the face grid.
    n_theta: Number of poloidal samples (endpoint-exclusive).

  Returns:
    ``B`` and FSA ``weights`` with shape ``(..., n_theta)``.
  """
  eps = constants.CONSTANTS.eps
  R_in = geo.R_in_face
  R_out = geo.R_out_face
  kappa = geo.elongation_face
  delta = jnp.clip(geo.delta_face, -0.999, 0.999)
  F = geo.F_face
  r = 0.5 * (R_out - R_in)
  R0 = 0.5 * (R_in + R_out)
  rho_n = geo.rho_face_norm

  dr_drhon = jnp.gradient(r, rho_n, axis=-1)
  dR0_drhon = jnp.gradient(R0, rho_n, axis=-1)
  dkappa_drhon = jnp.gradient(kappa, rho_n, axis=-1)
  ddelta_drhon = jnp.gradient(delta, rho_n, axis=-1)
  dr_drhon_safe = jnp.where(jnp.abs(dr_drhon) < eps, 1.0, dr_drhon)
  dR0_dr = dR0_drhon / dr_drhon_safe
  dkappa_dr = dkappa_drhon / dr_drhon_safe
  ddelta_dr = ddelta_drhon / dr_drhon_safe
  dx_dr = ddelta_dr / jnp.sqrt(jnp.maximum(1.0 - delta**2, eps))

  theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
  sinth = jnp.sin(theta)
  costh = jnp.cos(theta)
  x = jnp.arcsin(delta)
  theta_eff = theta + x[..., jnp.newaxis] * sinth

  R = R0[..., jnp.newaxis] + r[..., jnp.newaxis] * jnp.cos(theta_eff)
  dR_dr = dR0_dr[..., jnp.newaxis] + jnp.cos(theta_eff) - (
      r[..., jnp.newaxis]
      * jnp.sin(theta_eff)
      * sinth
      * dx_dr[..., jnp.newaxis]
  )
  dR_dth = (
      -r[..., jnp.newaxis]
      * (1.0 + x[..., jnp.newaxis] * costh)
      * jnp.sin(theta_eff)
  )
  dZ_dr = sinth * (kappa + r * dkappa_dr)[..., jnp.newaxis]
  dZ_dth = (kappa * r)[..., jnp.newaxis] * costh
  jacobian = dR_dr * dZ_dth - dR_dth * dZ_dr
  jacobian_abs = jnp.maximum(jnp.abs(jacobian), eps)
  l_theta = jnp.sqrt(dR_dth**2 + dZ_dth**2)
  grad_r = l_theta / jacobian_abs
  # NEO flux-surface-average weight √g dθ = R |∂(R,Z)/∂(r,θ)| dθ.
  weights = R * jacobian_abs

  B_tor = jnp.abs(F)[..., jnp.newaxis] / jnp.maximum(R, eps)
  q_safe = jnp.maximum(jnp.abs(q_face), eps)
  dpsi_drhon = 2.0 * geo.Phi_face[..., -1] * rho_n / q_safe
  dpsi_dr = dpsi_drhon / dr_drhon_safe
  B_pol = (
      jnp.abs(dpsi_dr)[..., jnp.newaxis]
      * grad_r
      / (2.0 * jnp.pi * jnp.maximum(R, eps))
  )
  B = jnp.sqrt(B_tor**2 + B_pol**2)
  return B, weights


# TODO(b/545148156): Add finite-orbit-width effects.
def calculate_f_trap(
    geo: geometry_lib.Geometry,
    f_trap_model: FTrapModel = 'sauter',
    q_face: array_typing.FloatVectorFace | None = None,
) -> array_typing.FloatVectorFace:
  """Calculates the effective trapped particle fraction.

  Args:
    geo: The magnetic geometry.
    f_trap_model: Which analytic/numerical approximation to use:

      * ``'sauter'``: O. Sauter, Fusion Engineering and Design 112 (2016)
        633-645, Eqs. 33+34 (includes triangularity via an effective
        inverse aspect ratio).
      * ``'simple'``: large-aspect-ratio circular approximation
        :math:`f_t = 1.46\\sqrt{\\epsilon} - 0.46\\epsilon`, using the
        midplane inverse aspect ratio ``geo.epsilon_face``.
      * ``'LinLiu'``: Lin-Liu & Miller weighted average of circular upper and
        lower bounds (Phys. Plasmas 2, 1666, 1995):
        :math:`f_t = 0.75 f_{tu} + 0.25 f_{tl}`, with
        :math:`f_{tu} = 1 - (1 - \\tfrac{3}{2}\\sqrt{\\epsilon} +
        \\tfrac{1}{2}\\epsilon^{3/2}) / \\sqrt{1-\\epsilon^2}` and
        :math:`f_{tl} = 1 - (1-\\epsilon)^2 /
        [\\sqrt{1-\\epsilon^2}\\,(1 + 1.46\\sqrt{\\epsilon} + 0.2\\epsilon)]`,
        using ``geo.epsilon_face``
      * ``'RABBIT'``: M. Weiland et al., Nucl. Fusion 58, 082032 (2018),
        :math:`f_t = 1.4624256\\sqrt{\\epsilon_\\mathrm{eff}} -
        0.46\\epsilon_\\mathrm{eff}^{3/2}`, with
        :math:`\\epsilon_\\mathrm{eff} = (R_\\mathrm{max} - R_0) / R_0`,
        where :math:`R_\\mathrm{max}` is the maximum major radius on the
        flux surface (``R_out``) and :math:`R_0` is the magnetic-axis
        major radius.
      * ``'numerical'``: Sauter PoP 1999 Eq. (12). If the geometry stores
        flux-surface :math:`|B|(\\theta)` and FSA weights (EQDSK), those
        are used directly. Otherwise a NEO/GACODE Miller surface is built
        with :math:`|B|=\\sqrt{(F/R)^2+B_p^2}` and Jacobian FSA weights,
        which requires ``q_face``.

    q_face: Safety factor on the face grid. Required for
      ``f_trap_model='numerical'`` when the geometry does not store
      equilibrium :math:`B(\\theta)` (sets the Miller :math:`B_p`
      amplitude).

  Returns:
    The effective trapped particle fraction.
  """
  if f_trap_model == 'sauter':
    epsilon_effective = (
        0.67
        * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face)
        * geo.epsilon_face
    )
    aa = (1.0 - geo.epsilon_face) / (1.0 + geo.epsilon_face)
    return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
        # On the magnetic axis, epsilon_effective is 0, in order to avoid a NaN
        # gradient we define the gradient at zero to be zero.
        1.0 + 2.0 * math_utils.sqrt_with_zero_gradient_at_zero(epsilon_effective)
    )
  if f_trap_model == 'simple':
    epsilon = geo.epsilon_face
    return 1.46 * jnp.sqrt(epsilon) - 0.46 * epsilon
  if f_trap_model == 'LinLiu':
    # Y. R. Lin-Liu and R. L. Miller, Phys. Plasmas 2, 1666 (1995).
    # Circular upper/lower bounds with the recommended 3/4–1/4 weighting.
    epsilon = geo.epsilon_face
    sqrt_eps = jnp.sqrt(epsilon)
    denom = jnp.sqrt(1.0 - epsilon**2)
    f_tu = 1.0 - (1.0 - 1.5 * sqrt_eps + 0.5 * epsilon * sqrt_eps) / denom
    f_tl = 1.0 - (1.0 - epsilon) ** 2 / (
        denom * (1.0 + 1.46 * sqrt_eps + 0.2 * epsilon)
    )
    return 0.75 * f_tu + 0.25 * f_tl
  if f_trap_model == 'RABBIT':
    # M. Weiland et al., Nucl. Fusion 58, 082032 (2018).
    # R_out is the maximum major radius on each flux surface; axis value is R_0.
    R_0 = geo.R_out_face[..., 0]
    epsilon_eff = (geo.R_out_face - R_0) / R_0
    return 1.4624256 * jnp.sqrt(epsilon_eff) - 0.46 * epsilon_eff**1.5
  if f_trap_model == 'numerical':
    B_eq = geo.B_surface_face
    w_eq = geo.fsa_weight_face
    if B_eq is not None and w_eq is not None:
      return trapped_fraction_from_B(B_eq, w_eq)
    if q_face is None:
      raise ValueError(
          "f_trap_model='numerical' requires q_face when the geometry "
          'does not store equilibrium B(θ) (Miller B_p amplitude).'
      )
    B, weights = _neo_miller_B_and_weights(geo, q_face)
    return trapped_fraction_from_B(B, weights)
  raise ValueError(
      f'Unknown f_trap_model={f_trap_model!r}. Supported values are'
      " 'sauter', 'simple', 'LinLiu', 'RABBIT', and 'numerical'."
  )


# TODO(b/428166775): currently we have two very similar implementations for
# nu_e_star. We should refactor this to have a single one in physics/collisions
def calculate_nu_e_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_e: array_typing.FloatVectorFace,
    T_e: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the electron collisionality, nu_e_star.

  This is the electron collisionality, defined as the ratio of the electron
  collision frequency to the bounce frequency. From Sauter PoP 1999 Eq. (18b).

  Args:
    q: Safety factor.
    geo: The geometry of the torus.
    n_e: Electron density [m^-3].
    T_e: Electron temperature [keV]. Converted to eV in the formula.
    Z_eff: Effective charge.
    log_lambda_ei: Electron-ion Coulomb logarithm.

  Returns:
    The electron collisionality.
  """
  return (
      6.921e-18
      * q
      * geo.R_major_profile_face
      * n_e
      * Z_eff
      * log_lambda_ei
      / (
          ((T_e * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )


def calculate_nu_i_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_i: array_typing.FloatVectorFace,
    T_i: array_typing.FloatVectorFace,
    Z_i: array_typing.FloatVectorFace,
    log_lambda_ii: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the ion collisionality, nu_i_star.

  From Sauter PoP 1999 Eq. (18c), with Z = Z_ion (not Z_eff) per the Sauter
  errata / neoclassical notes. For NEO multi-species plasmas this matches
  ``nui_star_S ∝ Z_ion^4 * dens_sum`` when ``n_i`` is set to
  ``dens_sum = Σ_s n_s`` over non-electrons (same symbol as in Sauter; the
  density argument is dens_sum, not bundled main-ion density alone).

  Args:
    q: Safety factor.
    geo: The geometry of the torus.
    n_i: Density in Sauter Eq. (18c) [m^-3]. Pass NEO ``dens_sum`` for
      multi-species bootstrap / Angioni; main-ion density for single-fluid
      uses (e.g. poloidal velocity).
    T_i: Ion temperature [keV].
    Z_i: Main ion charge (Sauter Eq. 18c / NEO ``z(is_ion)``).
    log_lambda_ii: Ion-ion Coulomb logarithm (Sauter Eq. 18e, also uses Z_ion).

  Returns:
    The ion collisionality.
  """
  return (
      4.9e-18
      * q
      * geo.R_major_profile_face
      * n_i
      * Z_i**4
      * log_lambda_ii
      / (
          ((T_i * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )


# Functions to calculate the neoclassical poloidal velocity.
def _calculate_neoclassical_k_neo(
    nu_star: array_typing.FloatScalar, epsilon: array_typing.FloatScalar
):
  """Calculates the neoclassical coefficient k_neo.

  Equation (6.135) from
  Hinton, F. L., & Hazeltine, R. D.,
  "Theory of plasma transport in toroidal confinement systems"
  Rev. Mod. Phys. 48(2), 239–308. (1976)
  https://doi.org/10.1103/RevModPhys.48.239

  Limits:
    - Banana regime (nu_star -> 0): ~1.17
    - Pfirsch-Schluter regime (nu_star -> inf): ~ -2.1

  Args:
    nu_star: The normalized ion collisionality.
    epsilon: The inverse aspect ratio.

  Returns:
    k_neo : The neoclassical coefficient.
  """
  # Calculate the first term (Banana-Plateau transition)
  # (1.17 - 0.35 * sqrt(nu)) / (1 + 0.7 * sqrt(nu))
  sqrt_nu = jnp.sqrt(nu_star)
  term1 = (1.17 - 0.35 * sqrt_nu) / (1.0 + 0.7 * sqrt_nu)

  # Calculate the second term (Pfirsch-Schluter driver)
  # 2.1 * nu^2 * epsilon^3
  ps_factor = (nu_star**2) * (epsilon**3)
  term2 = 2.1 * ps_factor

  # Calculate the final denominator (Switching function)
  # 1 + nu^2 * epsilon^3
  denominator = 1.0 + ps_factor

  return (term1 - term2) / denominator


# TODO(b/381199010): Implement alternative Sauter-based k_neo calculation.
# See Sauter (1999) Eq. 17a-17b


@jax.jit
def calculate_poloidal_velocity(
    T_i: cell_variable.CellVariable,
    n_i: array_typing.FloatVectorFace,
    q: array_typing.FloatVectorFace,
    Z_i: array_typing.FloatVectorFace,
    B_tor: array_typing.FloatVectorFace,
    B_total_squared: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0,
) -> cell_variable.CellVariable:
  """Computes the neoclassical ion poloidal velocity profile.

  Implementing eq.33 from
  Y. B. Kim , P. H. Diamond , R. J. Groebner.
  "Neoclassical poloidal and toroidal rotation in tokamaks"
  Phys. Fluids B 3, 2050–2060 (1991)
  https://doi.org/10.1063/1.859671

  Eq. 33 can be simplified to the following form in SI units:
  v_pol = k_neo * (dT/dr) * (B_tor / <B^2>) / (Z * e)

  Args:
    T_i: Ion temperature as a cell variable [keV].
    n_i: Ion density on the face grid [m^-3].
    q: Safety factor on the face grid.
    Z_i: Main ion charge on the face grid.
    B_tor: Toroidal magnetic field on the face grid [T].
    B_total_squared: Total magnetic field (toroidal + poloidal) on the face grid
      [T].
    geo: Geometry
    poloidal_velocity_multiplier: A multiplier to apply to the poloidal
      velocity.

  Returns:
    v_pol : Poloidal velocity profile [m/s].
  """
  # Note: all computations are performed on the face grid.

  T_i_face = T_i.face_value()
  epsilon = geo.epsilon_face

  # Calculate Neoclassical Coefficient k_i
  # Sauter Eqs. (18c) and (18e) use Z_ion, not Z_eff.
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i_face,  # pyrefly: ignore[bad-argument-type]
      n_i,  # pyrefly: ignore[bad-argument-type]
      Z_i,  # pyrefly: ignore[bad-argument-type]
  )
  nu_i_star = calculate_nu_i_star(
      q=q,
      geo=geo,
      n_i=n_i,
      T_i=T_i_face,  # pyrefly: ignore[bad-argument-type]
      Z_i=Z_i,
      log_lambda_ii=log_lambda_ii,
  )
  k_neo = _calculate_neoclassical_k_neo(nu_i_star, epsilon)

  # Calculate Radial Temperature Gradient (dT/dr)
  grad_Ti = (
      T_i.face_grad(
          x=geo.r_mid, x_left=geo.r_mid_face[0], x_right=geo.r_mid_face[-1]
      )
      * constants.CONSTANTS.keV_to_J
  )  # [J/m]

  # Calculate Poloidal Velocity
  # v_pol = k_i * (dT/dr) * (B_tor / <B^2>) / (Z * e)
  B_total_squared_safe = jnp.maximum(B_total_squared, constants.CONSTANTS.eps)
  v_pol = (
      k_neo
      * grad_Ti
      * (B_tor / B_total_squared_safe)
      / (constants.CONSTANTS.q_e * Z_i)
  )

  v_pol = poloidal_velocity_multiplier * v_pol

  return cell_variable.CellVariable(
      value=geometry_lib.face_to_cell(v_pol),
      face_centers=geo.rho_face_norm,
      right_face_constraint=v_pol[-1],
      right_face_grad_constraint=None,
  )


def _safe_log_grad(
    profile: cell_variable.CellVariable,
) -> array_typing.FloatVectorFace:
  """``∇ln`` of a face profile, safe when the value vanishes."""
  return math_utils.safe_divide(
      num=profile.face_grad(),
      denom=profile.face_value(),
      eps=constants.CONSTANTS.eps,
  )


def calculate_analytic_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    ion_species: Sequence[IonSpeciesProfiles],
    psi: cell_variable.CellVariable,
    geo: geometry_lib.Geometry,
    L31: array_typing.FloatVectorFace,
    L32: array_typing.FloatVectorFace,
    L34: array_typing.FloatVectorFace,
    alpha: array_typing.FloatVectorFace,
) -> bootstrap_current_base.BootstrapCurrent:
  """Shared analytic bootstrap current using NEO multi-species drives.

  Follows NEO ``compute_Sauter`` drive assembly (Sauter PoP 1999 fits for
  coefficients; Redl uses the same drive structure with Redl L-coeffs):

  * ``L31``: sum over electrons and all ion/impurity species of
    ``n_s T_s (∇ln n_s + ∇ln T_s)``
  * ``L32``: electron temperature gradient only
  * ``L34 * α``: sum over ion/impurity species of ``p_s ∇ln T_s``

  Each ion species contributes with its own ``T_s`` and ``∇ln T_s``.

  Args:
    bootstrap_multiplier: A multiplier for the bootstrap current.
    n_e: Electron density profile.
    T_e: Electron temperature profile.
    p_e: Electron pressure profile.
    ion_species: Thermal ion/impurity ``(n, T)`` profiles (no electrons).
    psi: Poloidal flux profile.
    geo: The magnetic geometry.
    L31: Neoclassical transport coefficient.
    L32: Neoclassical transport coefficient.
    L34: Neoclassical transport coefficient.
    alpha: Neoclassical coefficient related to ion-ion collisions.

  Returns:
    The bootstrap current profile.
  """
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = p_e.face_value()

  dpsi_drnorm = psi.face_grad()
  dlnne_drnorm = n_e.face_grad() / n_e.face_value()
  dlnte_drnorm = T_e.face_grad() / T_e.face_value()

  # Per-species ion drives: Σ_s p_s ∇ln n_s and Σ_s p_s ∇ln T_s.
  # Safe-divide so a vanishing species density or temperature does not
  # produce 0/0 or Inf.
  ion_density_drive = jnp.zeros_like(pe)
  ion_temperature_drive = jnp.zeros_like(pe)
  for species in ion_species:
    p_s = make_thermal_pressure(species.n, species.T).face_value()
    ion_density_drive = ion_density_drive + p_s * _safe_log_grad(species.n)
    ion_temperature_drive = ion_temperature_drive + p_s * _safe_log_grad(
        species.T
    )

  global_coeff = prefactor[1:] / dpsi_drnorm[1:]  # pyrefly: ignore[bad-index]
  global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])

  # Electrons: L31 * pe * (∇ln ne + ∇ln Te) + L32 * pe * ∇ln Te
  # Ions: L31 * Σ_s p_s (∇ln n_s + ∇ln T_s) + α L34 * Σ_s p_s ∇ln T_s
  j_parallel_bootstrap_face = global_coeff * (
      L31 * pe * dlnne_drnorm
      + (L31 + L32) * pe * dlnte_drnorm
      + L31 * ion_density_drive
      + (L31 + alpha * L34) * ion_temperature_drive
  )
  j_parallel_bootstrap = geometry_lib.face_to_cell(j_parallel_bootstrap_face)

  return bootstrap_current_base.BootstrapCurrent(
      j_parallel_bootstrap=j_parallel_bootstrap,  # pyrefly: ignore[bad-argument-type]
      j_parallel_bootstrap_face=j_parallel_bootstrap_face,
  )
