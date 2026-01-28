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
from typing import Annotated, Literal
import numpy as np
import pydantic
from torax._src import constants
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions


# pylint: disable=invalid-name
class CheaseConfig(base.BaseGeometryConfig):
  """Pydantic model for the CHEASE geometry.

  Attributes:
    geometry_type: Always set to 'chease'.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
  """

  geometry_type: Annotated[Literal['chease'], torax_pydantic.TIME_INVARIANT] = (
      'chease'
  )
  geometry_directory: Annotated[str | None, torax_pydantic.TIME_INVARIANT] = (
      None
  )
  Ip_from_parameters: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True
  geometry_file: str = 'iterhybrid.mat2cols'
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:
    intermediates = _construct_intermediates_from_chease(
        geometry_directory=self.geometry_directory,
        geometry_file=self.geometry_file,
        Ip_from_parameters=self.Ip_from_parameters,
        face_centers=self.get_face_centers(),
        R_major=self.R_major,
        a_minor=self.a_minor,
        B_0=self.B_0,
        hires_factor=self.hires_factor,
    )

    return standard_geometry.build_standard_geometry(intermediates)


# pylint: disable=invalid-name


def _construct_intermediates_from_chease(
    geometry_directory: str | None,
    geometry_file: str,
    Ip_from_parameters: bool,
    face_centers: np.ndarray,
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
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
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
      face_centers=face_centers,
      hires_factor=hires_factor,
      diverted=None,
      connection_length_target=None,
      connection_length_divertor=None,
      angle_of_incidence_target=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
      z_magnetic_axis=None,
  )
