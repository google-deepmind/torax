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


from __future__ import annotations

import numpy as np
from torax.geometry import geometry
from torax.torax_pydantic import torax_pydantic


# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name
def build_circular_geometry(
    n_rho: int,
    elongation_LCFS: float,
    Rmaj: float,
    Rmin: float,
    B0: float,
    hires_fac: int,
) -> geometry.Geometry:
  """Constructs a circular Geometry instance used for testing only.

  Args:
    n_rho: Radial grid points (num cells)
    elongation_LCFS: Elongation at last closed flux surface.
    Rmaj: major radius (R) in meters
    Rmin: minor radius (a) in meters
    B0: Toroidal magnetic field on axis [T]
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A Geometry instance.
  """
  # circular geometry assumption of r/Rmin = rho_norm, the normalized
  # toroidal flux coordinate.
  drho_norm = 1.0 / n_rho
  # Define mesh (Slab Uniform 1D with Jacobian = 1)
  mesh = torax_pydantic.Grid1D.construct(nx=n_rho, dx=drho_norm)
  # toroidal flux coordinate (rho) at boundary (last closed flux surface)
  rho_b = np.asarray(Rmin)

  # normalized and unnormalized toroidal flux coordinate (rho)
  # on face and cell grids. See fvm documentation and paper for details on
  # face and cell grids.
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers
  rho_face = rho_face_norm * rho_b
  rho = rho_norm * rho_b

  Rmaj = np.array(Rmaj)
  B0 = np.array(B0)

  # Define toroidal flux
  Phi = np.pi * B0 * rho**2
  Phi_face = np.pi * B0 * rho_face**2

  # Elongation profile.
  # Set to be a linearly increasing function from 1 to elongation_LCFS, which
  # is the elongation value at the last closed flux surface, set in config.
  elongation = 1 + rho_norm * (elongation_LCFS - 1)
  elongation_face = 1 + rho_face_norm * (elongation_LCFS - 1)

  # Volume in elongated circular geometry is given by:
  # V = 2*pi^2*R*rho^2*elongation
  # S = pi*rho^2*elongation

  volume = 2 * np.pi**2 * Rmaj * rho**2 * elongation
  volume_face = 2 * np.pi**2 * Rmaj * rho_face**2 * elongation_face
  area = np.pi * rho**2 * elongation
  area_face = np.pi * rho_face**2 * elongation_face

  # V' = dV/drnorm for volume integrations
  # \nabla V = 4*pi^2*R*rho*elongation
  #   + V * (elongation_param - 1) / elongation / rho_b
  # vpr = \nabla V * rho_b
  vpr = 4 * np.pi**2 * Rmaj * rho * elongation * rho_b + volume / elongation * (
      elongation_LCFS - 1
  )
  vpr_face = (
      4 * np.pi**2 * Rmaj * rho_face * elongation_face * rho_b
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
  g2 = g1 / Rmaj**2
  g2_face = g1_face / Rmaj**2

  # g3: <1/R^2> (done without a elongation correction)
  # <1/R^2> =
  # 1/2pi*int_0^2pi (1/(Rmaj+r*cosx)^2)dx =
  # 1/( Rmaj^2 * (1 - (r/Rmaj)^2)^3/2 )
  g3 = 1 / (Rmaj**2 * (1 - (rho / Rmaj) ** 2) ** (3.0 / 2.0))
  g3_face = 1 / (Rmaj**2 * (1 - (rho_face / Rmaj) ** 2) ** (3.0 / 2.0))

  # simplifying assumption for now, for J=R*B/(R0*B0)
  J = np.ones(len(rho))
  J_face = np.ones(len(rho_face))
  # simplified (constant) version of the F=B*R function
  F = np.ones(len(rho)) * Rmaj * B0
  F_face = np.ones(len(rho_face)) * Rmaj * B0

  # Using an approximation where:
  # g2g3_over_rhon = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc elongation assumption, which leads to more
  # reasonable values for g2g3_over_rhon through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rhon will be used.

  g2g3_over_rhon = 4 * np.pi**2 * vpr * g3 / (J * Rmaj)
  g2g3_over_rhon_face = 4 * np.pi**2 * vpr_face * g3_face / (J_face * Rmaj)

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current, which is
  # the only option for ad-hoc circular geometry.
  rho_hires_norm = np.linspace(0, 1, n_rho * hires_fac)
  rho_hires = rho_hires_norm * rho_b

  Rout = Rmaj + rho
  Rout_face = Rmaj + rho_face

  Rin = Rmaj - rho
  Rin_face = Rmaj - rho_face

  # assumed elongation profile on hires grid
  elongation_hires = 1 + rho_hires_norm * (elongation_LCFS - 1)

  volume_hires = 2 * np.pi**2 * Rmaj * rho_hires**2 * elongation_hires
  area_hires = np.pi * rho_hires**2 * elongation_hires

  # V' = dV/drnorm for volume integrations on hires grid
  vpr_hires = (
      4 * np.pi**2 * Rmaj * rho_hires * elongation_hires * rho_b
      + volume_hires / elongation_hires * (elongation_LCFS - 1)
  )
  # S' = dS/drnorm for area integrals on hires grid
  spr_hires = (
      2 * np.pi * rho_hires * elongation_hires * rho_b
      + area_hires / elongation_hires * (elongation_LCFS - 1)
  )

  g3_hires = 1 / (Rmaj**2 * (1 - (rho_hires / Rmaj) ** 2) ** (3.0 / 2.0))
  F_hires = np.ones(len(rho_hires)) * B0 * Rmaj
  g2g3_over_rhon_hires = 4 * np.pi**2 * vpr_hires * g3_hires * B0 / F_hires

  return geometry.Geometry(
      # Set the standard geometry params.
      geometry_type=geometry.GeometryType.CIRCULAR,
      torax_mesh=mesh,
      Phi=Phi,
      Phi_face=Phi_face,
      Rmaj=Rmaj,
      Rmin=rho_b,
      B0=B0,
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
      g2g3_over_rhon=g2g3_over_rhon,
      g2g3_over_rhon_face=g2g3_over_rhon_face,
      g2g3_over_rhon_hires=g2g3_over_rhon_hires,
      F=F,
      F_face=F_face,
      F_hires=F_hires,
      Rin=Rin,
      Rin_face=Rin_face,
      Rout=Rout,
      Rout_face=Rout_face,
      # Set the circular geometry-specific params.
      elongation=elongation,
      elongation_face=elongation_face,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
      _z_magnetic_axis=np.asarray(0.0),
  )
