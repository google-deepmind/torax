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
"""Functions for loading and representing a CHEASE geometry."""
import numpy as np
from torax._src import constants
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
# pylint: disable=invalid-name


def from_chease(
    geometry_directory: str | None,
    geometry_file: str,
    Ip_from_parameters: bool,
    n_rho: int,
    R_major: float,
    a_minor: float,
    B_0: float,
    hires_factor: int,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from a CHEASE file.

  Args:
    geometry_directory: Directory where to find the CHEASE file describing the
      magnetic geometry. If None, then it defaults to another dir. See
      implementation.
    geometry_file: CHEASE file name.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    n_rho: Radial grid points (num cells)
    R_major: major radius (R) in meters. CHEASE geometries are normalized, so
      this is used as an unnormalization factor.
    a_minor: minor radius (a) in meters
    B_0: Toroidal magnetic field on axis [T].
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A StandardGeometry instance based on the input file. This can then be
    used to build a StandardGeometry by passing to `build_standard_geometry`.
  """
  chease_data = geometry_loader.load_geo_data(
      geometry_directory, geometry_file, geometry_loader.GeometrySource.CHEASE
  )

  # Prepare variables from CHEASE to be interpolated into our simulation
  # grid. CHEASE variables are normalized. Need to unnormalize them with
  # reference values poloidal flux and CHEASE-internal-calculated plasma
  # current.
  # Also, CHEASE is COCOS=02, which means psi_torax = 2π * psi_chease
  psiunnormfactor = R_major**2 * B_0 * 2 * np.pi
  psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor
  Ip_chease = (
      chease_data['Ipprofile'] / constants.CONSTANTS.mu_0 * R_major * B_0
  )

  # toroidal flux
  Phi = (chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] * R_major) ** 2 * B_0 * np.pi

  # midplane radii
  R_in_chease = chease_data['R_INBOARD'] * R_major
  R_out_chease = chease_data['R_OUTBOARD'] * R_major
  # toroidal field flux function
  F = chease_data['T=RBphi'] * R_major * B_0

  int_dl_over_Bp = (
      chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] * R_major / B_0
  )
  flux_surf_avg_1_over_R = chease_data['<1/R>profile'] / R_major
  flux_surf_avg_1_over_R2 = chease_data['<1/R**2>'] / R_major**2
  # COCOS > 10: <|\nabla \psi|> = 2\pi<R B_p>
  flux_surf_avg_grad_psi2_over_R2 = (
      chease_data['<Bp**2>'] * B_0**2 * (4 * np.pi**2)
  )
  flux_surf_avg_grad_psi = (
      chease_data['<|grad(psi)|>'] * psiunnormfactor / R_major
  )
  flux_surf_avg_grad_psi2 = (
      chease_data['<|grad(psi)|**2>'] * psiunnormfactor**2 / R_major**2
  )
  flux_surf_avg_B2 = chease_data['<B**2>'] * B_0**2
  flux_surf_avg_1_over_B2 = chease_data['<1/B**2>'] / B_0**2

  rhon = np.sqrt(Phi / Phi[-1])
  vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.CHEASE,
      Ip_from_parameters=Ip_from_parameters,
      R_major=np.array(R_major),
      a_minor=np.array(a_minor),
      B_0=np.array(B_0),
      psi=psi,
      Ip_profile=Ip_chease,
      Phi=Phi,
      R_in=R_in_chease,
      R_out=R_out_chease,
      F=F,
      int_dl_over_Bp=int_dl_over_Bp,
      flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
      flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
      flux_surf_avg_grad_psi2_over_R2=flux_surf_avg_grad_psi2_over_R2,
      flux_surf_avg_grad_psi=flux_surf_avg_grad_psi,
      flux_surf_avg_grad_psi2=flux_surf_avg_grad_psi2,
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      delta_upper_face=chease_data['delta_upper'],
      delta_lower_face=chease_data['delta_bottom'],
      elongation=chease_data['elongation'],
      vpr=vpr,
      n_rho=n_rho,
      hires_factor=hires_factor,
      diverted=None,
      connection_length_target=None,
      connection_length_divertor=None,
      target_angle_of_incidence=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
      z_magnetic_axis=None,
  )
