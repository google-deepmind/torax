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
"""Classes for representing a circular geometry."""

from collections.abc import Mapping
import functools
from typing import Annotated, Literal, Self
import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import trapped_fraction
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class CircularConfig(base.GeometryConfigBase):
  """Ad-hoc analytical circular geometry model with elongation corrections.

  Intended for testing purposes only.

  Attributes:
    model_name: Always set to `'circular'`.
    R_major: Major radius, defined as the midplane average of the LCFS radius,
      $R_major = (R_max + R_min) / 2$ [m].
    a_minor: Minor radius, defined as half the horizontal width of the LCFS,
      $a_minor = (R_max - R_min) / 2$ [m]. Must satisfy `a_minor <= R_major`.
    B_0: Vacuum toroidal magnetic field at `R_major` [T].
    elongation_LCFS: Sets the plasma elongation at the last closed flux surface
      used for volume, area, and q-profile corrections, and in transport models.
  """

  @property
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction.TrappedFractionSource]:
    return frozenset({trapped_fraction.TrappedFractionSource.SAUTER})

  model_name: Annotated[
      Literal['circular'],
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = 'circular'
  R_major: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(6.2)
  )
  a_minor: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(2.0)
  )
  B_0: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(5.3)
  elongation_LCFS: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.72)
  )

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    # Check the union of configured time points across the piecewise-linear
    # scalar inputs to guarantee a_minor <= R_major for all t.
    time_points = sorted(
        set(self.R_major.time)
        | set(self.a_minor.time)
        | set(self.B_0.time)
        | set(self.elongation_LCFS.time)
    )
    for t_k in time_points:
      if not float(self.R_major.get_value(t_k)) >= float(
          self.a_minor.get_value(t_k)
      ):
        raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def _build_geometry(
      self,
  ) -> geometry.Geometry | Mapping[float, geometry.Geometry]:
    scalars = (self.R_major, self.a_minor, self.B_0, self.elongation_LCFS)
    if (
        all(len(s.time) == 1 for s in scalars)
        and len({float(s.time[0]) for s in scalars}) == 1
    ):
      t = float(self.R_major.time[0])
      return _build_circular_geometry(
          face_centers=self.get_face_centers(),
          elongation_LCFS=float(self.elongation_LCFS.get_value(t)),
          R_major=float(self.R_major.get_value(t)),
          a_minor=float(self.a_minor.get_value(t)),
          B_0=float(self.B_0.get_value(t)),
          hires_factor=self.hires_factor,
      )

    time_points = sorted(
        set(self.R_major.time)
        | set(self.a_minor.time)
        | set(self.B_0.time)
        | set(self.elongation_LCFS.time)
    )
    return {
        float(t_k): _build_circular_geometry(
            face_centers=self.get_face_centers(),
            elongation_LCFS=float(self.elongation_LCFS.get_value(t_k)),
            R_major=float(self.R_major.get_value(t_k)),
            a_minor=float(self.a_minor.get_value(t_k)),
            B_0=float(self.B_0.get_value(t_k)),
            hires_factor=self.hires_factor,
        )
        for t_k in time_points
    }


@functools.partial(jax.jit, static_argnames=('hires_factor',))
def _build_circular_geometry(
    face_centers: array_typing.FloatVectorFace,
    elongation_LCFS: array_typing.FloatScalar,
    R_major: array_typing.FloatScalar,
    a_minor: array_typing.FloatScalar,
    B_0: array_typing.FloatScalar,
    hires_factor: int,
) -> geometry.Geometry:
  """Constructs a circular Geometry instance.

  Args:
    face_centers: Coordinates of face centers in normalized rho [0, 1].
    elongation_LCFS: Elongation at last closed flux surface.
    R_major: major radius (R) in meters
    a_minor: minor radius (a) in meters
    B_0: Vacuum toroidal magnetic field at `R_major` [T]
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A Geometry instance.
  """
  dtype = jax_utils.get_dtype()
  rho_b = jnp.asarray(a_minor, dtype=dtype)
  R_major = jnp.asarray(R_major, dtype=dtype)
  B_0 = jnp.asarray(B_0, dtype=dtype)
  elongation_LCFS = jnp.asarray(elongation_LCFS, dtype=dtype)

  # circular geometry assumption of r/a_minor = rho_norm, the normalized
  # toroidal flux coordinate.
  # normalized and unnormalized toroidal flux coordinate (rho)
  # on face and cell grids. See fvm documentation and paper for details on
  # face and cell grids.
  rho_face_norm = jnp.asarray(face_centers, dtype=dtype)
  rho_norm = geometry.face_to_cell(rho_face_norm)
  rho_face = rho_face_norm * rho_b
  rho = rho_norm * rho_b

  # Define toroidal flux
  Phi = jnp.pi * B_0 * rho**2
  Phi_face = jnp.pi * B_0 * rho_face**2

  # Elongation profile.
  # Set to be a linearly increasing function from 1 to elongation_LCFS, which
  # is the elongation value at the last closed flux surface, set in config.
  elongation = 1 + rho_norm * (elongation_LCFS - 1)
  elongation_face = 1 + rho_face_norm * (elongation_LCFS - 1)

  # Volume in elongated circular geometry is given by:
  # V = 2*pi^2*R*rho^2*elongation
  # S = pi*rho^2*elongation

  volume = 2 * jnp.pi**2 * R_major * rho**2 * elongation
  volume_face = 2 * jnp.pi**2 * R_major * rho_face**2 * elongation_face
  area = jnp.pi * rho**2 * elongation
  area_face = jnp.pi * rho_face**2 * elongation_face

  # V' = dV/drnorm for volume integrations
  # \nabla V = 4*pi^2*R*rho*elongation
  #   + V * (elongation_param - 1) / elongation / rho_b
  # vpr = \nabla V * rho_b
  vpr = (
      4 * jnp.pi**2 * R_major * rho * elongation * rho_b
      + volume / elongation * (elongation_LCFS - 1)
  )
  vpr_face = (
      4 * jnp.pi**2 * R_major * rho_face * elongation_face * rho_b
      + volume_face / elongation_face * (elongation_LCFS - 1)
  )
  # S' = dS/drnorm for area integrals on cell grid
  spr = 2 * jnp.pi * rho * elongation * rho_b + area / elongation * (
      elongation_LCFS - 1
  )
  spr_face = (
      2 * jnp.pi * rho_face * elongation_face * rho_b
      + area_face / elongation_face * (elongation_LCFS - 1)
  )

  delta_face = jnp.zeros_like(rho_face)

  # Geometry variables for general geometry form of transport equations.
  # With circular geometry approximation.

  # g0: <\nabla V>
  g0 = vpr / rho_b
  g0_face = vpr_face / rho_b

  # g1: <(\nabla V)^2>
  g1 = vpr**2 / rho_b**2
  g1_face = vpr_face**2 / rho_b**2

  # g2: <(\nabla V)^2 / R^2>
  g2 = g1 / R_major**2
  g2_face = g1_face / R_major**2

  # g3: <1/R^2> (done without a elongation correction)
  # <1/R^2> =
  # 1/2pi*int_0^2pi (1/(R_major+r*cosx)^2)dx =
  # 1/( R_major^2 * (1 - (r/R_major)^2)^3/2 )
  g3 = 1 / (R_major**2 * (1 - (rho / R_major) ** 2) ** (3.0 / 2.0))
  g3_face = 1 / (R_major**2 * (1 - (rho_face / R_major) ** 2) ** (3.0 / 2.0))

  # simplifying assumption for now, for J=R*B/(R0*B_0)
  J = jnp.ones_like(rho)
  J_face = jnp.ones_like(rho_face)
  # simplified (constant) version of the F=B*R function
  F = jnp.ones_like(rho) * R_major * B_0
  F_face = jnp.ones_like(rho_face) * R_major * B_0

  # Using an approximation where:
  # g2g3_over_rhon = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc elongation assumption, which leads to more
  # reasonable values for g2g3_over_rhon through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rhon will be used.

  g2g3_over_rhon = 4 * jnp.pi**2 * vpr * g3 / (J * R_major)
  g2g3_over_rhon_face = 4 * jnp.pi**2 * vpr_face * g3_face / (J_face * R_major)

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current, which is
  # the only option for ad-hoc circular geometry.
  rho_hires_norm = geometry.increase_grid_resolution(
      rho_face_norm, hires_factor
  )
  rho_hires = rho_hires_norm * rho_b

  R_out = R_major + rho
  R_out_face = R_major + rho_face

  R_in = R_major - rho
  R_in_face = R_major - rho_face

  # assumed elongation profile on hires grid
  elongation_hires = 1 + rho_hires_norm * (elongation_LCFS - 1)

  volume_hires = 2 * jnp.pi**2 * R_major * rho_hires**2 * elongation_hires
  area_hires = jnp.pi * rho_hires**2 * elongation_hires

  # V' = dV/drnorm for volume integrations on hires grid
  vpr_hires = (
      4 * jnp.pi**2 * R_major * rho_hires * elongation_hires * rho_b
      + volume_hires / elongation_hires * (elongation_LCFS - 1)
  )
  # S' = dS/drnorm for area integrals on hires grid
  spr_hires = (
      2 * jnp.pi * rho_hires * elongation_hires * rho_b
      + area_hires / elongation_hires * (elongation_LCFS - 1)
  )

  # Analytical expressions for  <1/B^2> (gm4) and <B^2> (gm5)
  epsilon = (R_out - R_in) / (R_out + R_in)
  epsilon_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
  trapped_fraction_face = trapped_fraction.calculate_sauter_trapped_fraction(
      epsilon=epsilon_face, delta=delta_face
  )
  gm4 = B_0**-2 * (1.0 + 1.5 * epsilon**2)
  gm4_face = B_0**-2 * (1.0 + 1.5 * epsilon_face**2)
  gm5 = B_0**2 / jnp.sqrt(1.0 - epsilon**2)
  gm5_face = B_0**2 / jnp.sqrt(1.0 - epsilon_face**2)

  g3_hires = 1 / (R_major**2 * (1 - (rho_hires / R_major) ** 2) ** (3.0 / 2.0))
  F_hires = jnp.ones_like(rho_hires) * B_0 * R_major
  g2g3_over_rhon_hires = 4 * jnp.pi**2 * vpr_hires * g3_hires * B_0 / F_hires

  return geometry.Geometry(
      # Set the standard geometry params.
      geometry_type=geometry.GeometryType.CIRCULAR,
      rho_face_norm=rho_face_norm,
      Phi=Phi,
      Phi_face=Phi_face,
      R_major=R_major,
      a_minor=rho_b,
      B_0=B_0,
      volume=volume,
      volume_face=volume_face,
      area=area,
      area_face=area_face,
      vpr=vpr,
      vpr_face=vpr_face,
      spr=spr,
      spr_face=spr_face,
      delta_face=delta_face,
      trapped_fraction_face=trapped_fraction_face,
      g0=g0,
      g0_face=g0_face,
      g1=g1,
      g1_face=g1_face,
      g2=g2,
      g2_face=g2_face,
      g3=g3,
      g3_face=g3_face,
      gm4=gm4,
      gm4_face=gm4_face,
      gm5=gm5,
      gm5_face=gm5_face,
      g2g3_over_rhon=g2g3_over_rhon,
      g2g3_over_rhon_face=g2g3_over_rhon_face,
      g2g3_over_rhon_hires=g2g3_over_rhon_hires,
      F=F,
      F_face=F_face,
      F_hires=F_hires,
      R_in=R_in,
      R_in_face=R_in_face,
      R_out=R_out,
      R_out_face=R_out_face,
      # Set the circular geometry-specific params.
      elongation=elongation,
      elongation_face=elongation_face,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      _z_magnetic_axis=jnp.asarray(0.0, dtype=dtype),
  )
