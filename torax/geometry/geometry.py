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

"""Classes for representing the problem geometry."""


from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum

import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax.torax_pydantic import torax_pydantic


def face_to_cell(face: chex.Array) -> chex.Array:
  """Infers cell values corresponding to a vector of face values.

  Simply a linear interpolation between face values.

  Args:
    face: An array containing face values.

  Returns:
    cell: An array containing cell values.
  """

  return 0.5 * (face[:-1] + face[1:])


@enum.unique
class GeometryType(enum.IntEnum):
  """Integer enum for geometry type.

  This type can be used within JAX expressions to access the geometry type
  without having to call isinstance.
  """

  CIRCULAR = 0
  CHEASE = 1
  FBT = 2
  EQDSK = 3


# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class Geometry:
  r"""Describes the magnetic geometry.

  Most users should default to using the StandardGeometry class, whether the
  source of their geometry comes from CHEASE, MEQ, EQDSK, etc.

  Properties work for both 1D radial arrays and 2D stacked arrays where the
  leading dimension is time.

  Attributes:
    geometry_type: Type of geometry model used. See `GeometryType` for options.
    torax_mesh: `Grid1D` object representing the radial mesh used by TORAX.
    Phi: Toroidal magnetic flux at each radial grid point [:math:`\mathrm{Wb}`].
    Phi_face: Toroidal magnetic flux at each radial face [:math:`\mathrm{Wb}`].
    Rmaj: Tokamak major radius (geometric center) [:math:`\mathrm{m}`].
    Rmin: Tokamak minor radius [:math:`\mathrm{m}`].
    B0: Vacuum toroidal magnetic field on axis [:math:`\mathrm{T}`].
    volume: Plasma volume enclosed by each flux surface on cell grid
      [:math:`\mathrm{m}^3`].
    volume_face: Plasma volume enclosed by each flux surface on face grid
      [:math:`\mathrm{m}^3`].
    area: Poloidal cross-sectional area of each flux surface on cell grid
      [:math:`\mathrm{m}^2`].
    area_face: Poloidal cross-sectional area of each flux surface on face grid
      [:math:`\mathrm{m}^2`].
    vpr: Derivative of plasma volume enclosed by each flux surface with respect
      to the normalized toroidal flux coordinate rho_norm on cell grid
      [:math:`\mathrm{m}^3`].
    vpr_face: Derivative of plasma volume enclosed by each flux surface with
      respect to the normalized toroidal flux coordinate rho_face_norm, on face
      grid [:math:`\mathrm{m}^3`].
    spr: Derivative of plasma surface area enclosed by each flux surface, with
      respect to the normalized toroidal flux coordinate rho_norm on cell grid
      [:math:`\mathrm{m}^2`]. Equal to vpr / (:math:`2 \pi` Rmaj).
    spr_face: Derivative of plasma surface area enclosed by each flux surface,
      with respect to the normalized toroidal flux coordinate rho_face_norm on
      face grid [:math:`\mathrm{m}^2`]. Equal to vpr_face / (:math:`2 \pi`
      Rmaj).
    spr_hires: Derivative of plasma surface area enclosed by each flux surface
      on a higher resolution grid, with respect to the normalized toroidal flux
      coordinate rho_norm. [:math:`\mathrm{m}^2`].
    rho_hires: Toroidal flux coordinate on a higher resolution grid
      [:math:`\mathrm{m}`].
    rho_hires_norm: Normalized toroidal flux coordinate on a higher resolution
      grid [dimensionless].
    g0: Flux surface averaged radial derivative of the plasma volume:
      :math:`\langle \nabla V \rangle` on cell grid [:math:`\mathrm{m}^2`].
    g0_face: Flux surface averaged :math:`\langle \nabla V \rangle` on the faces
      [:math:`\mathrm{m}^2`].
    g1: Flux surface averaged :math:`\langle (\nabla V)^2 \rangle` on cell grid
      [:math:`\mathrm{m}^4`].
    g1_face: Flux surface averaged :math:`\langle (\nabla V)^2 \rangle` on the
      faces [:math:`\mathrm{m}^4`].
    g2: Flux surface averaged :math:`\langle (\nabla V)^2 / R^2 \rangle` on cell
      grid [:math:`\mathrm{m}^2`], where R is the major radius along the flux
      surface being averaged.
    g2_face: Flux surface averaged :math:`\langle (\nabla V)^2 / R^2 \rangle` on
      the faces [:math:`\mathrm{m}^2`].
    g3: Flux surface averaged :math:`\langle 1 / R^2 \rangle` on cell grid
      [:math:`\mathrm{m}^{-2}`].
    g3_face: Flux surface averaged :math:`\langle 1 / R^2 \rangle` on the faces
      [:math:`\mathrm{m}^{-2}`].
    g2g3_over_rhon: Ratio of g2g3 to the normalized toroidal flux coordinate
      rho_norm on cell grid [dimensionless].
    g2g3_over_rhon_face: Ratio of g2g3 to the normalized toroidal flux
      coordinate rho_norm on face grid [dimensionless].
    g2g3_over_rhon_hires: Ratio of g2g3 to the normalized toroidal flux
      coordinate rho_norm on a higher resolution grid [dimensionless].
    F: Toroidal field flux function, :math:`F \equiv RB_\phi` on cell grid,
      where :math:`R` is major radius, and :math:`B_\phi` is the toroidal
      magnetic field [:math:`\mathrm{T m}`].
    F_face: Toroidal field flux function, :math:`F \equiv RB_\phi` on face grid
      [:math:`\mathrm{T m}`].
    F_hires: Toroidal field flux function, :math:`F \equiv RB_\phi` on the high
      resolution grid [:math:`\mathrm{T m}`].
    Rin: Radius of the flux surface at the inboard side at midplane
      [:math:`\mathrm{m}`] on cell grid. Inboard side is defined as the minimum
      radial extent of the flux surface.
    Rin_face: Radius of the flux surface at the inboard side at midplane
      [:math:`\mathrm{m}`] on face grid.
    Rout: Radius of the flux surface at the outboard side at midplane
      [:math:`\mathrm{m}`] on cell grid. Outboard side is defined as the maximum
      radial extent of the flux surface.
    Rout_face: Radius of the flux surface at the outboard side at midplane
      [:math:`\mathrm{m}`] on face grid.
    delta_face: Average of upper and lower triangularity of each flux surface at
      the faces [dimensionless]. Upper triangularity is defined as (Rmaj_local -
      R_upper) / Rmin_local, where Rmaj_local = (Rout+Rin)/2, Rmin_local =
      (Rout-Rin)/2, and R_upper is the radial location of the upper extent of
      the flux surface. Lower triangularity is defined as (Rmaj_local - R_lower)
      / Rmin_local, where R_lower is the radial location of the lower extent of
      the flux surface.
    elongation: Plasma elongation profile on cell grid [dimensionless].
      Elongation is defined as (Z_upper - Z_lower) / (2.0 * Rmin_local), where
      Z_upper and Z_lower are the Z coordinates of the upper and lower extent of
      the flux surface.
    elongation_face: Plasma elongation profile on face grid [dimensionless].
    Phibdot: Time derivative of the toroidal magnetic flux
      [:math:`\mathrm{Wb/s}`]. Calculated across a time interval using ``Phi``
      from the Geometry objects at time t and t + dt. See
      ``torax.orchestration.step_function`` for more details.
    _z_magnetic_axis: Vertical position of the magnetic axis
      [:math:`\mathrm{m}`].
  """

  geometry_type: GeometryType
  torax_mesh: torax_pydantic.Grid1D
  Phi: chex.Array
  Phi_face: chex.Array
  Rmaj: chex.Array
  Rmin: chex.Array
  B0: chex.Array
  volume: chex.Array
  volume_face: chex.Array
  area: chex.Array
  area_face: chex.Array
  vpr: chex.Array
  vpr_face: chex.Array
  spr: chex.Array
  spr_face: chex.Array
  delta_face: chex.Array
  elongation: chex.Array
  elongation_face: chex.Array
  g0: chex.Array
  g0_face: chex.Array
  g1: chex.Array
  g1_face: chex.Array
  g2: chex.Array
  g2_face: chex.Array
  g3: chex.Array
  g3_face: chex.Array
  g2g3_over_rhon: chex.Array
  g2g3_over_rhon_face: chex.Array
  g2g3_over_rhon_hires: chex.Array
  F: chex.Array
  F_face: chex.Array
  F_hires: chex.Array
  Rin: chex.Array
  Rin_face: chex.Array
  Rout: chex.Array
  Rout_face: chex.Array
  spr_hires: chex.Array
  rho_hires_norm: chex.Array
  rho_hires: chex.Array
  Phibdot: chex.Array
  _z_magnetic_axis: chex.Array | None

  @property
  def q_correction_factor(self) -> chex.Numeric:
    """Ad-hoc fix for non-physical circular geometry model.

    Set such that q(r=a) = 3 for standard ITER parameters.
    """
    return jnp.where(
        self.geometry_type == GeometryType.CIRCULAR.value,
        1.25,
        1,
    )

  @property
  def rho_norm(self) -> chex.Array:
    r"""Normalized toroidal flux coordinate on cell grid [dimensionless]."""
    return self.torax_mesh.cell_centers

  @property
  def rho_face_norm(self) -> chex.Array:
    r"""Normalized toroidal flux coordinate on face grid [dimensionless]."""
    return self.torax_mesh.face_centers

  @property
  def drho_norm(self) -> chex.Array:
    r"""Grid size for rho_norm [dimensionless]."""
    return jnp.array(self.torax_mesh.dx)

  @property
  def rho_face(self) -> chex.Array:
    r"""Toroidal flux coordinate on face grid :math:`\mathrm{m}`."""
    return self.rho_face_norm * jnp.expand_dims(self.rho_b, axis=-1)

  @property
  def rho(self) -> chex.Array:
    r"""Toroidal flux coordinate on cell grid :math:`\mathrm{m}`.

    The toroidal flux coordinate is defined as
    :math:`\rho=\sqrt{\frac{\Phi}{\pi B_0}}`, where :math:`\Phi` is the
    toroidal flux enclosed by the flux surface, and :math:`B_0` the magnetic
    field on the magnetic axis.
    """
    return self.rho_norm * jnp.expand_dims(self.rho_b, axis=-1)

  @property
  def rmid(self) -> chex.Array:
    """Midplane radius of the plasma [m], defined as (Rout-Rin)/2."""
    return (self.Rout - self.Rin) / 2

  @property
  def rmid_face(self) -> chex.Array:
    """Midplane radius of the plasma on the face grid [m]."""
    return (self.Rout_face - self.Rin_face) / 2

  @property
  def drho(self) -> chex.Array:
    """Grid size for rho [m]."""
    return self.drho_norm * self.rho_b

  @property
  def rho_b(self) -> chex.Array:
    """Toroidal flux coordinate [m] at boundary (LCFS)."""
    return jnp.sqrt(self.Phib / np.pi / self.B0)

  @property
  def Phib(self) -> chex.Array:
    r"""Toroidal flux at boundary (LCFS) :math:`\mathrm{Wb}`."""
    return self.Phi_face[..., -1]

  @property
  def g1_over_vpr(self) -> chex.Array:
    r"""g1/vpr [:math:`\mathrm{m}`]."""
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    r"""g1/vpr**2 [:math:`\mathrm{m}^{-2}`]."""
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    """g0_face/vpr_face [:math:`m^{-1}`], equal to 1/rho_b on-axis."""
    # Calculate the bulk of the array (excluding the first element)
    # to avoid division by zero.
    bulk = self.g0_face[..., 1:] / self.vpr_face[..., 1:]
    first_element = jnp.ones_like(self.rho_b) / self.rho_b
    # Concatenate to handle both 1D (no leading dim) and 2D cases
    return jnp.concatenate(
        [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1
    )

  @property
  def g1_over_vpr_face(self) -> jax.Array:
    r"""g1_face/vpr_face [:math:`\mathrm{m}`]. Zero on-axis."""
    bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:]
    first_element = jnp.zeros_like(self.rho_b)
    return jnp.concatenate(
        [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1
    )

  @property
  def g1_over_vpr2_face(self) -> jax.Array:
    """g1_face/vpr_face**2 [:math:`m^{-2}`], equal to 1/rho_b**2 on-axis."""
    bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:] ** 2
    first_element = jnp.ones_like(self.rho_b) / self.rho_b**2
    return jnp.concatenate(
        [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1
    )

  def z_magnetic_axis(self) -> chex.Numeric:
    """z position of magnetic axis [m]."""
    z_magnetic_axis = self._z_magnetic_axis
    if z_magnetic_axis is not None:
      return z_magnetic_axis
    else:
      raise ValueError('Geometry does not have a z magnetic axis.')


def stack_geometries(geometries: Sequence[Geometry]) -> Geometry:
  """Batch together a sequence of geometries.

  Args:
    geometries: A sequence of geometries to stack. The geometries must have the
      same mesh, geometry type.

  Returns:
    A Geometry object, where each array attribute has an additional
    leading axis (e.g. for the time dimension) compared to each Geometry in
    the input sequence.
  """
  if not geometries:
    raise ValueError('No geometries provided.')
  # Ensure that all geometries have same mesh and are of same type.
  first_geo = geometries[0]
  torax_mesh = first_geo.torax_mesh
  geometry_type = first_geo.geometry_type
  for geometry in geometries[1:]:
    if geometry.torax_mesh != torax_mesh:
      raise ValueError('All geometries must have the same mesh.')
    if geometry.geometry_type != geometry_type:
      raise ValueError('All geometries must have the same geometry type.')

  stacked_data = {}
  for field in dataclasses.fields(first_geo):
    field_name = field.name
    field_value = getattr(first_geo, field_name)
    # Stack stackable fields. Save first geo's value for non-stackable fields.
    if isinstance(field_value, chex.Array):
      field_values = [getattr(geo, field_name) for geo in geometries]
      stacked_data[field_name] = jnp.stack(field_values)
    else:
      stacked_data[field_name] = field_value
  # Create a new object with the stacked data with the same class (i.e.
  # could be child classes of Geometry)
  return first_geo.__class__(**stacked_data)


def update_geometries_with_Phibdot(
    *,
    dt: chex.Numeric,
    geo_t: Geometry,
    geo_t_plus_dt: Geometry,
) -> tuple[Geometry, Geometry]:
  """Update Phibdot in the geometry dataclasses used in the time interval.

  Phibdot is used in calc_coeffs to calcuate terms related to time-dependent
  geometry. It should be set to be the same for geo_t and geo_t_plus_dt for
  each given time interval. This means that geo_t_plus_dt.Phibdot will not
  necessarily be the same as the geo_t.Phibdot at the next time step.

  Args:
    dt: Time step duration.
    geo_t: The geometry of the torus during this time step of the simulation.
    geo_t_plus_dt: The geometry of the torus during the next time step of the
      simulation.

  Returns:
    Tuple containing:
      - The geometry of the torus during this time step of the simulation.
      - The geometry of the torus during the next time step of the simulation.
  """
  Phibdot = (geo_t_plus_dt.Phib - geo_t.Phib) / dt
  geo_t = dataclasses.replace(geo_t, Phibdot=Phibdot)
  geo_t_plus_dt = dataclasses.replace(geo_t_plus_dt, Phibdot=Phibdot)
  return geo_t, geo_t_plus_dt
