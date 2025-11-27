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
from typing import Annotated
from typing import Literal
import numpy as np
import pydantic
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions


# pylint: disable=invalid-name
class CircularConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the circular geometry config.

  Attributes:
    geometry_type: Always set to 'circular'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
    elongation_LCFS: Sets the plasma elongation used for volume, area and
      q-profile corrections.
  """

  geometry_type: Annotated[
      Literal['circular'], torax_pydantic.TIME_INVARIANT
  ] = 'circular'
  n_rho: Annotated[pydantic.PositiveInt, torax_pydantic.TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3
  elongation_LCFS: pydantic.PositiveFloat = 1.72

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def build_geometry(self) -> geometry.Geometry:
    return build_circular_geometry(
        n_rho=self.n_rho,
        elongation_LCFS=self.elongation_LCFS,
        R_major=self.R_major,
        a_minor=self.a_minor,
        B_0=self.B_0,
        hires_factor=self.hires_factor,
    )


def build_circular_geometry(
    n_rho: int,
    elongation_LCFS: float,
    R_major: float,
    a_minor: float,
    B_0: float,
    hires_factor: int,
) -> geometry.Geometry:
  """Constructs a circular Geometry instance used for testing only.

  Args:
    n_rho: Radial grid points (num cells)
    elongation_LCFS: Elongation at last closed flux surface.
    R_major: major radius (R) in meters
    a_minor: minor radius (a) in meters
    B_0: Toroidal magnetic field on axis [T]
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A Geometry instance.
  """
  # circular geometry assumption of r/a_minor = rho_norm, the normalized
  # toroidal flux coordinate.
  # Define mesh (Slab Uniform 1D with Jacobian = 1)
  mesh = torax_pydantic.Grid1D(nx=n_rho,)
  # toroidal flux coordinate (rho) at boundary (last closed flux surface)
  rho_b = np.asarray(a_minor)

  # normalized and unnormalized toroidal flux coordinate (rho)
  # on face and cell grids. See fvm documentation and paper for details on
  # face and cell grids.
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers
  rho_face = rho_face_norm * rho_b
  rho = rho_norm * rho_b

  R_major = np.array(R_major)
  B_0 = np.array(B_0)

  # Define toroidal flux
  Phi = np.pi * B_0 * rho**2
  Phi_face = np.pi * B_0 * rho_face**2

  # Elongation profile.
  # Set to be a linearly increasing function from 1 to elongation_LCFS, which
  # is the elongation value at the last closed flux surface, set in config.
  elongation = 1 + rho_norm * (elongation_LCFS - 1)
  elongation_face = 1 + rho_face_norm * (elongation_LCFS - 1)

  # Volume in elongated circular geometry is given by:
  # V = 2*pi^2*R*rho^2*elongation
  # S = pi*rho^2*elongation

  volume = 2 * np.pi**2 * R_major * rho**2 * elongation
  volume_face = 2 * np.pi**2 * R_major * rho_face**2 * elongation_face
  area = np.pi * rho**2 * elongation
  area_face = np.pi * rho_face**2 * elongation_face

  # V' = dV/drnorm for volume integrations
  # \nabla V = 4*pi^2*R*rho*elongation
  #   + V * (elongation_param - 1) / elongation / rho_b
  # vpr = \nabla V * rho_b
  vpr = (
      4 * np.pi**2 * R_major * rho * elongation * rho_b
      + volume / elongation * (elongation_LCFS - 1)
  )
  vpr_face = (
      4 * np.pi**2 * R_major * rho_face * elongation_face * rho_b
      + volume_face / elongation_face * (elongation_LCFS - 1)
  )
  # pylint: disable=invalid-name
  # S' = dS/drnorm for area integrals on cell grid
  spr = 2 * np.pi * rho * elongation * rho_b + area / elongation * (
      elongation_LCFS - 1
  )
  spr_face = (
      2 * np.pi * rho_face * elongation_face * rho_b
      + area_face / elongation_face * (elongation_LCFS - 1)
  )

  delta_face = np.zeros(len(rho_face))

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
  J = np.ones(len(rho))
  J_face = np.ones(len(rho_face))
  # simplified (constant) version of the F=B*R function
  F = np.ones(len(rho)) * R_major * B_0
  F_face = np.ones(len(rho_face)) * R_major * B_0

  # Using an approximation where:
  # g2g3_over_rhon = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc elongation assumption, which leads to more
  # reasonable values for g2g3_over_rhon through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rhon will be used.

  g2g3_over_rhon = 4 * np.pi**2 * vpr * g3 / (J * R_major)
  g2g3_over_rhon_face = 4 * np.pi**2 * vpr_face * g3_face / (J_face * R_major)

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current, which is
  # the only option for ad-hoc circular geometry.
  rho_hires_norm = np.linspace(0, 1, n_rho * hires_factor)
  rho_hires = rho_hires_norm * rho_b

  R_out = R_major + rho
  R_out_face = R_major + rho_face

  R_in = R_major - rho
  R_in_face = R_major - rho_face

  # assumed elongation profile on hires grid
  elongation_hires = 1 + rho_hires_norm * (elongation_LCFS - 1)

  volume_hires = 2 * np.pi**2 * R_major * rho_hires**2 * elongation_hires
  area_hires = np.pi * rho_hires**2 * elongation_hires

  # V' = dV/drnorm for volume integrations on hires grid
  vpr_hires = (
      4 * np.pi**2 * R_major * rho_hires * elongation_hires * rho_b
      + volume_hires / elongation_hires * (elongation_LCFS - 1)
  )
  # S' = dS/drnorm for area integrals on hires grid
  spr_hires = (
      2 * np.pi * rho_hires * elongation_hires * rho_b
      + area_hires / elongation_hires * (elongation_LCFS - 1)
  )

  # Analytical expressions for  <1/B^2> (gm4) and <B^2> (gm5)
  epsilon = (R_out - R_in) / (R_out + R_in)
  epsilon_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
  gm4 = B_0**-2 * (1.0 + 1.5 * epsilon**2)
  gm4_face = B_0**-2 * (1.0 + 1.5 * epsilon_face**2)
  gm5 = B_0**2 / np.sqrt(1.0 - epsilon**2)
  gm5_face = B_0**2 / np.sqrt(1.0 - epsilon_face**2)

  g3_hires = 1 / (R_major**2 * (1 - (rho_hires / R_major) ** 2) ** (3.0 / 2.0))
  F_hires = np.ones(len(rho_hires)) * B_0 * R_major
  g2g3_over_rhon_hires = 4 * np.pi**2 * vpr_hires * g3_hires * B_0 / F_hires

  return geometry.Geometry(
      # Set the standard geometry params.
      geometry_type=geometry.GeometryType.CIRCULAR,
      torax_mesh=mesh,
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
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phi_b_dot=np.asarray(0.0),
      _z_magnetic_axis=np.asarray(0.0),
  )
