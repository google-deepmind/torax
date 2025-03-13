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

"""Calculations related to derived quantities from poloidal flux (psi).

Functions:
    - calc_q: Calculates the q-profile (q).
    - calc_jtot: Calculate flux-surface-averaged toroidal current density.
    - calc_s: Calculates magnetic shear (s).
    - calc_s_rmid: Calculates magnetic shear (s), using midplane r as radial
      coordinate.
    - calc_Wpol: Calculates total magnetic energy (Wpol).
    - calc_li3: Calculates normalized internal inductance li3 (ITER convention).
    - calc_q95: Calculates the q-profile at 95% of the normalized poloidal flux.
    - calculate_psi_grad_constraint_from_Ip_tot: Calculates the gradient
      constraint on the poloidal flux (psi) from Ip.
    - _calc_bpol2: Calculates square of poloidal field (Bp).
"""
import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax.fvm import cell_variable
from torax.fvm import convection_terms
from torax.fvm import diffusion_terms
from torax.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def calc_q_face(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> chex.Array:
  """Calculates the q-profile on the face grid given poloidal flux (psi)."""
  # iota is standard terminology for 1/q
  inv_iota = jnp.abs(
      (2 * geo.Phib * geo.rho_face_norm[1:]) / psi.face_grad()[1:]
  )

  # Use L'Hôpital's rule to calculate iota on-axis, with psi_face_grad()[0]=0.
  inv_iota0 = jnp.expand_dims(
      jnp.abs((2 * geo.Phib * geo.drho_norm) / psi.face_grad()[1]), 0
  )

  q_face = jnp.concatenate([inv_iota0, inv_iota])
  return q_face * geo.q_correction_factor


def calc_jtot(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> tuple[chex.Array, chex.Array, chex.Array]:
  """Calculate flux-surface-averaged toroidal current density from poloidal flux.

  Calculation based on jtot = dI/dS

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    jtot: total current density [A/m2] on cell grid
    jtot_face: total current density [A/m2] on face grid
    Ip_profile_face: cumulative total plasma current profile [A] on face grid
  """

  # pylint: disable=invalid-name
  Ip_profile_face = (
      psi.face_grad()
      * geo.g2g3_over_rhon_face
      * geo.F_face
      / geo.Phib
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu0)
  )

  Ip_profile = (
      psi.grad()
      * geo.g2g3_over_rhon
      * geo.F
      / geo.Phib
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu0)
  )

  dI_drhon_face = jnp.gradient(Ip_profile_face, geo.rho_face_norm)
  dI_drhon = jnp.gradient(Ip_profile, geo.rho_norm)

  jtot_bulk = dI_drhon[1:] / geo.spr[1:]
  jtot_face_bulk = dI_drhon_face[1:] / geo.spr_face[1:]

  # Extrapolate the axis term from the bulk term due to strong sensitivities
  # of near-axis numerical derivatives. Set zero boundary condition on-axis
  jtot_axis = jtot_bulk[0] - (jtot_bulk[1] - jtot_bulk[0])

  jtot = jnp.concatenate([jnp.array([jtot_axis]), jtot_bulk])
  jtot_face = jnp.concatenate([jnp.array([jtot_axis]), jtot_face_bulk])

  return jtot, jtot_face, Ip_profile_face


def calc_s_face(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates magnetic shear on the face grid from poloidal flux (psi)."""

  # iota (1/q) should have a /2*Phib but we drop it since will cancel out in
  # the s calculation.
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))

  # on-axis iota_scaled from L'Hôpital's rule = dpsi_face_grad / drho_norm
  # Using expand_dims to make it compatible with jnp.concatenate
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )

  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])

  s_face = (
      -geo.rho_face_norm
      * jnp.gradient(iota_scaled, geo.rho_face_norm)
      / iota_scaled
  )

  return s_face


def calc_s_rmid(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates magnetic shear (s) from poloidal flux (psi).

  Version taking the derivative of iota with respect to the midplane r,
  in line with expectations from circular-derived models like QuaLiKiz.

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    s_face: Magnetic shear, on the face grid.
  """

  # iota (1/q) should have a /2*Phib but we drop it since will cancel out in
  # the s calculation.
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))

  # on-axis iota_scaled from L'Hôpital's rule = dpsi_face_grad / drho_norm
  # Using expand_dims to make it compatible with jnp.concatenate
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )

  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])

  rmid_face = (geo.Rout_face - geo.Rin_face) * 0.5

  s_face = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled

  return s_face


def _calc_bpol2(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  r"""Calculates square of poloidal field (Bp) from poloidal flux (psi).

  An identity for the poloidal magnetic field is:
  :math:`B_p = 1/R \partial \psi / \partial \rho (\nabla \rho \times e_\phi)`

  Where :math:`e_\phi` is the unit vector pointing in the toroidal direction.

  Args:
    geo: Torus geometry.
    psi: Poloidal flux.

  Returns:
    bpol2_face: Square of poloidal magnetic field, on the face grid.
  """
  bpol2_bulk = (
      (psi.face_grad()[1:] / (2 * jnp.pi)) ** 2
      * geo.g2_face[1:]
      / geo.vpr_face[1:] ** 2
  )
  bpol2_axis = jnp.array([0.0])
  bpol2_face = jnp.concatenate([bpol2_axis, bpol2_bulk])
  return bpol2_face


def calc_Wpol(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  """Calculates total magnetic energy (Wpol) from poloidal flux (psi)."""
  bpol2 = _calc_bpol2(geo, psi)
  Wpol = _trapz(bpol2 * geo.vpr_face, geo.rho_face_norm) / (
      2 * constants.CONSTANTS.mu0
  )
  return Wpol


def calc_li3(
    Rmaj: jax.Array,
    Wpol: jax.Array,
    Ip_total: jax.Array,
) -> jax.Array:
  """Calculates li3 based on a formulation using Wpol.

  Normalized internal inductance is defined as:
  li = <Bpol^2>_V / <Bpol^2>_LCFS where <>_V is a volume average and <>_LCFS is
  the average at the last closed flux surface.

  We use the ITER convention for normalized internal inductance defined as:
  li3 = 2*V*<Bpol^2>_V / (mu0^2 Ip^2*Rmaj) = 4 * Wpol / (mu0 Ip^2*Rmaj)

  Ip (total plasma current) enters through the integral form of Ampere's law.
  Since Wpol also corresponds to a volume integral of the poloidal field, we
  can define li3 with respect to Wpol.

  Args:
    Rmaj: Major radius.
    Wpol: Total magnetic energy.
    Ip_total: Total plasma current.

  Returns:
    li3: Normalized internal inductance, ITER convention.
  """
  return 4 * Wpol / (constants.CONSTANTS.mu0 * Ip_total**2 * Rmaj)


def calc_q95(
    psi_norm_face: array_typing.ArrayFloat,
    q_face: array_typing.ArrayFloat,
) -> array_typing.ScalarFloat:
  """Calculates q95 from the q profile and the normalized poloidal flux.

  Args:
    psi_norm_face: normalized poloidal flux
    q_face: safety factor on the face grid

  Returns:
    q95: q at 95% of the normalized poloidal flux
  """
  q95 = jnp.interp(0.95, psi_norm_face, q_face)

  return q95


def calculate_psi_grad_constraint_from_Ip_tot(
    Ip_tot: array_typing.ScalarFloat,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates the gradient constraint on the poloidal flux (psi) from Ip."""
  return (
      Ip_tot
      * 1e6
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )


def calculate_psidot_from_psi_sources(
    *,
    psi_sources: array_typing.ArrayFloat,
    sigma: array_typing.ArrayFloat,
    sigma_face: array_typing.ArrayFloat,
    resistivity_multiplier: float,
    psi: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates psidot (loop voltage) from the sum of the psi sources."""

  # Calculate transient term
  consts = constants.CONSTANTS
  toc_psi = (
      1.0
      / resistivity_multiplier
      * geo.rho_norm
      * sigma
      * consts.mu0
      * 16
      * jnp.pi**2
      * geo.Phib**2
      / geo.F**2
  )
  # Calculate diffusion term coefficient
  d_face_psi = geo.g2g3_over_rhon_face
  # Add phibdot terms to poloidal flux convection
  v_face_psi = (
      -8.0
      * jnp.pi**2
      * consts.mu0
      * geo.Phibdot
      * geo.Phib
      * sigma_face
      * geo.rho_face_norm**2
      / geo.F_face**2
  )

  # Add effective phibdot poloidal flux source term
  ddrnorm_sigma_rnorm2_over_f2 = jnp.gradient(
      sigma * geo.rho_norm**2 / geo.F**2, geo.rho_norm
  )

  psi_sources += (
      -8.0
      * jnp.pi**2
      * consts.mu0
      * geo.Phibdot
      * geo.Phib
      * ddrnorm_sigma_rnorm2_over_f2
  )

  diffusion_mat, diffusion_vec = diffusion_terms.make_diffusion_terms(
      d_face_psi, psi
  )

  # Set the psi convection term for psidot used in ohmic power, always with
  # the default 'ghost' mode. Impact of different modes would mildly impact
  # Ohmic power at the LCFS which has negligible impact on simulations.
  # Allowing it to be configurable introduces more complexity in the code by
  # needing to pass in the mode from the static_runtime_params across multiple
  # functions.
  conv_mat, conv_vec = convection_terms.make_convection_terms(
      v_face_psi, d_face_psi, psi
  )

  c_mat = diffusion_mat + conv_mat
  c = diffusion_vec + conv_vec

  c += psi_sources

  psidot = (jnp.dot(c_mat, psi.value) + c) / toc_psi

  return psidot
