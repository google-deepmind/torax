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
"""Classes for representing an EQDSK geometry."""

import json
import logging
from typing import Annotated, Any, Literal
from unittest import mock

import contourpy
import eqdsk
import eqdsk.file
import eqdsk.tools
import numpy as np
import numpy.typing as npt
import pydantic
import scipy
from torax._src import constants
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# Inject `npt` into eqdsk.file's runtime namespace to prevent Pydantic from
# raising `PydanticUndefinedAnnotation: name 'npt' is not defined.`
# Import of `npt` in upstream eqdsk is gated by TYPE_CHECKING, which is not
# compatible with Pydantic type validation.
# TODO(b/524519841): Upstream this fix to eqdsk.
eqdsk.file.npt = npt

# COCOS convention that TORAX translates all EQDSK geometries to.
_TORAX_EQDSK_COCOS = 11


# pylint: disable=invalid-name
class EQDSKConfig(base.BaseGeometryConfig):
  """Pydantic model for the EQDSK geometry.

  Attributes:
    cocos: COCOS coordinate convention of the EQDSK file, specified as an
      integer in the range 1-8 or 11-18 inclusive.
    geometry_file: Name of the EQDSK file in the geometry directory.
    eqdsk_object: In-memory EQDSKInterface object.
    geometry_type: Always set to 'eqdsk'.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    n_surfaces: Number of surfaces for which flux surface averages are
      calculated.
    last_surface_factor: Multiplication factor of the boundary poloidal flux,
      used for the contour defining geometry terms at the LCFS on the TORAX
      grid. Needed to avoid divergent integrations in diverted geometries.
  """

  cocos: torax_pydantic.COCOSInt = ...
  geometry_file: str | None = None
  eqdsk_object: eqdsk.EQDSKInterface | None = None
  geometry_type: Annotated[Literal['eqdsk'], torax_pydantic.TIME_INVARIANT] = (
      'eqdsk'
  )
  geometry_directory: Annotated[str | None, torax_pydantic.TIME_INVARIANT] = (
      None
  )
  Ip_from_parameters: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True
  n_surfaces: pydantic.PositiveInt = 100
  last_surface_factor: torax_pydantic.OpenUnitInterval = 0.99

  @pydantic.field_validator('eqdsk_object', mode='before')
  @classmethod
  def _eqdskinterface_before_validator(
      cls, x: eqdsk.EQDSKInterface | dict[str, Any] | None,
  ) -> eqdsk.EQDSKInterface | None:
    """Convert input to an EQDSKInterface object or None."""
    if x is None:
      return None
    elif isinstance(x, eqdsk.EQDSKInterface):
      return x
    elif isinstance(x, dict):
      json_str = json.dumps(x)
      # Mock all file openers to return the json string, allowing
      # EQDSKInterface.from_file to deserialize in-memory without disk I/O.
      # TODO(b/519910776): Use eqdsk library's json_reader when ready.
      # https://github.com/Fusion-Power-Plant-Framework/eqdsk/issues/140
      m = mock.mock_open(read_data=json_str)
      with mock.patch('builtins.open', m), mock.patch('io.open', m):
        return eqdsk.EQDSKInterface.from_file('eqdsk.json', no_cocos=True)
    else:
      raise ValueError(f'Expected EQDSKInterface or dict, but got {type(x)}')

  @pydantic.field_serializer('eqdsk_object', return_type=dict[str, Any] | None)
  @classmethod
  def _eqdskinterface_serializer(
      cls, obj: eqdsk.EQDSKInterface | None,
  ) -> dict[str, Any] | None:
    if obj is None:
      return None
    # Use eqdsk library's json_writer to serialize in-memory without disk I/O.
    json_str = eqdsk.tools.json_writer(obj.to_dict())
    return json.loads(json_str)

  @pydantic.model_validator(mode='after')
  def _validate_model(self) -> typing_extensions.Self:
    if self.geometry_file is None and self.eqdsk_object is None:
      raise ValueError(
          "Either 'geometry_file' or 'eqdsk_object' must be provided."
      )
    if self.geometry_file is not None and self.eqdsk_object is not None:
      raise ValueError(
          "Cannot provide both 'geometry_file' and 'eqdsk_object'."
      )
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:
    intermediates = _construct_intermediates_from_eqdsk(
        geometry_directory=self.geometry_directory,
        geometry_file=self.geometry_file,
        eqdsk_object=self.eqdsk_object,
        Ip_from_parameters=self.Ip_from_parameters,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
        cocos=self.cocos,
        n_surfaces=self.n_surfaces,
        last_surface_factor=self.last_surface_factor,
    )
    return standard_geometry.build_standard_geometry(intermediates)


def _construct_intermediates_from_eqdsk(
    geometry_directory: str | None,
    geometry_file: str | None,
    eqdsk_object: eqdsk.EQDSKInterface | None,
    hires_factor: int,
    Ip_from_parameters: bool,
    face_centers: np.ndarray,
    n_surfaces: int,
    last_surface_factor: float,
    cocos: int,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from EQDSK.

  This method constructs a StandardGeometryIntermediates object from an EQDSK
  file or object. It calculates flux surface averages based on the EQDSK
  geometry 2D psi mesh.

  Args:
    geometry_directory: Directory where to find the EQDSK file describing the
      magnetic geometry. If None, then it defaults to another dir. See
      implementation.
    geometry_file: EQDSK file name. Only used if eqdsk_object is None.
    eqdsk_object: EQDSKInterface object. If None, then the geometry file is
      loaded from the geometry directory.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    Ip_from_parameters: If True, then Ip is taken from the config and the values
      in the Geometry are rescaled.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    n_surfaces: Number of surfaces for which flux surface averages are
      calculated.
    last_surface_factor: Multiplication factor of the boundary poloidal flux,
      used for the contour defining geometry terms at the LCFS on the TORAX
      grid. Needed to avoid divergent integrations in diverted geometries.
    cocos: COCOS convention of the EQDSK file, specified as an integer between
      1-8 or 11-18 inclusive.

  Returns:
    A StandardGeometryIntermediates instance based on the input file or object.
    This can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """

  def calculate_area(x, z):
    """Gauss-shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)."""
    z_shifted = np.roll(z, -1)
    x_shifted = np.roll(x, -1)
    return 0.5 * np.abs(np.sum(x * z_shifted - z * x_shifted))

  # --------------------------- #
  # ---- 1. Load the eqdsk ---- #
  # --------------------------- #
  if cocos in [2, 4, 6, 8, 12, 14, 16, 18]:
    logging.warning(
        'User-specified COCOS %s for EQDSK file has a (R, Z, phi)'
        ' coordinate system (sigma_RphiZ = -1), but the EQDSK format'
        ' specifies a (R, phi, Z) coordinate system (sigma_RphiZ = +1). This'
        ' may result in unexpected behaviour.',
        cocos,
    )
  if eqdsk_object is not None:
    eq_dict = _convert_eqdsk_object_to_cocos11_dict(cocos, eqdsk_object)
  elif geometry_file is not None:
    eq_dict = geometry_loader.load_geo_data(
        geometry_dir=geometry_directory,
        geometry_file=geometry_file,
        geometry_source=geometry_loader.GeometrySource.EQDSK,
        cocos=cocos,
    )
  else:
    raise ValueError(
        'Either geometry_file or eqdsk_object must be provided to build EQDSK'
        ' geometry.'
    )

  # Check TORAX sign convention.
  eq_dict = _enforce_torax_sign_convention(eq_dict)
  _validate_torax_sign_convention(eq_dict)

  # Reference geometry terms
  # TODO(b/375696414): deal with updown asymmetric cases.
  # R_major taken as Rgeo (LCFS R_major)
  R_major = (eq_dict['xbdry'].max() + eq_dict['xbdry'].min()) / 2.0
  a_minor = (eq_dict['xbdry'].max() - eq_dict['xbdry'].min()) / 2.0
  # Vacuum toroidal magnetic field at R_major. EQDSK vacuum field, bcentre, is
  # defined at an arbitrary reference radius, xcentre. To get B at R_major,
  # use B_vacuum ∝ 1/R.

  # See EQDSKInterface in https://github.com/Fusion-Power-Plant-Framework/eqdsk
  B_0 = eq_dict['bcentre'] * eq_dict['xcentre'] / R_major
  Raxis = eq_dict['xmag']
  Zaxis = eq_dict['zmag']
  Btor_axis = eq_dict['fpol'][0] / eq_dict['xmag']

  # 1D psi grid, with psi(axis) = 0
  psi_1dgrid = np.linspace(
      0.0, eq_dict['psibdry'] - eq_dict['psimag'], eq_dict['nx']
  )

  # 2D X-Z grid
  X_1D = np.linspace(
      eq_dict['xgrid1'], eq_dict['xgrid1'] + eq_dict['xdim'], eq_dict['nx']
  )
  Z_1D = np.linspace(
      eq_dict['zmid'] - eq_dict['zdim'] / 2,
      eq_dict['zmid'] + eq_dict['zdim'] / 2,
      eq_dict['nz'],
  )
  X, Z = np.meshgrid(X_1D, Z_1D, indexing='ij')
  Xlcfs, Zlcfs = eq_dict['xbdry'], eq_dict['zbdry']

  # 2D psi grid, with psi(axis) = 0
  psi_2dgrid = eq_dict['psi'] - eq_dict['psimag']

  # Mask for the region inside the LCFS
  # i.e. Xlcfs.min() < X < Xlcfs.max() and Zlcfs.min() < Z < Zlcfs.max()
  offset = 0.01
  mask = (
      (X > Xlcfs.min() - offset)
      & (X < Xlcfs.max() + offset)
      & (Z > Zlcfs.min() - offset)
      & (Z < Zlcfs.max() + offset)
  )
  masked_psi_2dgrid = np.ma.masked_where(~mask, psi_2dgrid)

  # --------------------------------------- #
  # ---- 2. Make flux surface contours ---- #
  # --------------------------------------- #
  psi_on_flux_surfaces = np.linspace(
      0,
      (eq_dict['psibdry'] - eq_dict['psimag']) * last_surface_factor,
      n_surfaces,
  )

  surfaces = []
  psi_contour_generator = contourpy.contour_generator(X, Z, masked_psi_2dgrid)

  # Skip magnetic axis since no contour is defined there.
  for _, _psi in enumerate(psi_on_flux_surfaces[1:]):
    vertices = psi_contour_generator.create_contour(_psi)
    if not vertices:
      raise ValueError(f"""
          Valid contour not found for EQDSK geometry for psi value {_psi}.
          Possible reason is too many surfaces requested.
          Try reducing n_surfaces from the current value of {n_surfaces}.
          """)
    x_surface, z_surface = vertices[0].T[0], vertices[0].T[1]
    surfaces.append((x_surface, z_surface))

  # ------------------------------------------------------------------ #
  # ---- 3. Interpolate everything onto the new flux surface grid ---- #
  # ------------------------------------------------------------------ #
  # Spline interpolator of 2D psi field defined on X-Z grid
  # This will later be evaluated on each flux surface
  psi_2dgrid_interpolator = scipy.interpolate.RectBivariateSpline(
      X_1D, Z_1D, psi_2dgrid, kx=3, ky=3, s=0
  )

  # Interpolate safety factor onto new flux-surface grid
  q_interpolator = scipy.interpolate.interp1d(
      psi_1dgrid, eq_dict['qpsi'], kind='cubic'
  )
  q_profile = q_interpolator(psi_on_flux_surfaces)

  # Interpolate toroidal field flux function onto new flux-surface grid
  F_interpolator = scipy.interpolate.interp1d(
      psi_1dgrid, eq_dict['fpol'], kind='cubic'
  )
  F = F_interpolator(psi_on_flux_surfaces)

  # ---------------------------------------------------------- #
  # ---- 4. Compute flux surface averages and 1D profiles ---- #
  # ---------------------------------------------------------- #
  # - Area, Volume, R_inboard, R_outboard
  # - FSA: <1/R^2>, <Bp^2>, <|grad(psi)|>, <|grad(psi)|^2>
  # - Toroidal plasma current
  # - Integral dl/Bp

  # Initialise arrays
  areas, volumes = np.empty(len(surfaces) + 1), np.empty(len(surfaces) + 1)
  R_inboard, R_outboard = np.empty(len(surfaces) + 1), np.empty(
      len(surfaces) + 1
  )
  flux_surf_avg_1_over_R = np.empty(len(surfaces) + 1)  # <1/R>
  flux_surf_avg_1_over_R2 = np.empty(len(surfaces) + 1)  # <1/R**2>
  flux_surf_avg_grad_psi2_over_R2 = np.empty(len(surfaces) + 1)  # <Bp**2>
  flux_surf_avg_grad_psi = np.empty(len(surfaces) + 1)  # <|grad(psi)|>
  flux_surf_avg_grad_psi2 = np.empty(len(surfaces) + 1)  # <|grad(psi)|**2>
  flux_surf_avg_B2 = np.empty(len(surfaces) + 1)  # <B**2>
  flux_surf_avg_1_over_B2 = np.empty(len(surfaces) + 1)  # <1/B**2>
  int_dl_over_Bp = np.empty(len(surfaces) + 1)  # int(Rdl / | grad(psi) |)
  Ip = np.empty(len(surfaces) + 1)  # Toroidal plasma current
  delta_upper_face = np.empty(len(surfaces) + 1)  # Upper face delta
  delta_lower_face = np.empty(len(surfaces) + 1)  # Lower face delta
  elongation = np.empty(len(surfaces) + 1)  # Elongation

  # Compute fsa for each surface
  # Note: surfaces is from psi[1:]
  for n, (x_surface, z_surface) in enumerate(surfaces):
    # Define line elements on which we will integrate
    surface_dl = np.sqrt(
        np.gradient(x_surface) ** 2 + np.gradient(z_surface) ** 2
    )

    # Calculate gradient of psi in 2D
    surface_dpsi_x = psi_2dgrid_interpolator.ev(x_surface, z_surface, dx=1)
    surface_dpsi_z = psi_2dgrid_interpolator.ev(x_surface, z_surface, dy=1)
    surface_abs_grad_psi = np.sqrt(surface_dpsi_x**2 + surface_dpsi_z**2)

    # B components
    # Poloidal field strength Bp = |grad(psi)| / 2piR (COCOS>10)
    surface_Bpol = surface_abs_grad_psi / (2 * np.pi * x_surface)
    # Toroidal field strength Btor = F/R
    # Note: F[n+1] is F on this flux surface
    surface_Btor = F[n + 1] / x_surface
    # B**2
    surface_B2 = surface_Bpol**2 + surface_Btor**2

    # Plasma current
    surface_int_bpol_dl = np.sum(surface_Bpol * surface_dl)

    # Flux surface averaged equilibrium terms
    # <1/R>, < 1/ R^2>, < | grad psi | >, < B_pol^2>, < | grad psi |^2 >
    # where FSA(G) = int (G dl / Bpol) / (int (dl / Bpol))
    surface_int_dl_over_bpol = np.sum(surface_dl / surface_Bpol)
    surface_FSA_int_one_over_r = (
        np.sum(1 / x_surface * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )
    surface_FSA_int_one_over_r2 = (
        np.sum(1 / x_surface**2 * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )
    surface_FSA_abs_grad_psi = (
        np.sum(surface_abs_grad_psi * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )
    surface_FSA_abs_grad_psi2 = (
        np.sum(surface_abs_grad_psi**2 * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )
    surface_FSA_abs_grad_psi2_over_R2 = (
        np.sum(
            surface_abs_grad_psi**2 / x_surface**2 * surface_dl / surface_Bpol
        )
        / surface_int_dl_over_bpol
    )

    # <B**2> and <1/B**2> terms
    surface_FSA_B2 = (
        np.sum(surface_B2 * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )
    surface_FSA_1_over_B2 = (
        np.sum(1 / surface_B2 * surface_dl / surface_Bpol)
        / surface_int_dl_over_bpol
    )

    # Volumes and areas
    area = calculate_area(x_surface, z_surface)
    volume = area * 2 * np.pi * R_major

    # Triangularity
    # (RMAJ - X_upperextent) / RMIN
    idx_upperextent = np.argmax(z_surface)
    idx_lowerextent = np.argmin(z_surface)

    R_major_local = (x_surface.max() + x_surface.min()) / 2.0
    a_minor_local = (x_surface.max() - x_surface.min()) / 2.0

    X_upperextent = x_surface[idx_upperextent]
    X_lowerextent = x_surface[idx_lowerextent]

    Z_upperextent = z_surface[idx_upperextent]
    Z_lowerextent = z_surface[idx_lowerextent]

    surface_delta_upper_face = (R_major_local - X_upperextent) / a_minor_local
    surface_delta_lower_face = (R_major_local - X_lowerextent) / a_minor_local

    # Insert computed values into arrays
    # Note: n is going from 0 to len(psi_on_flux_surfaces)-2, so we
    # index by n+1 to fill fsa_arrays[1:]
    areas[n + 1] = area
    volumes[n + 1] = volume
    R_inboard[n + 1] = x_surface.min()
    R_outboard[n + 1] = x_surface.max()
    int_dl_over_Bp[n + 1] = surface_int_dl_over_bpol
    flux_surf_avg_1_over_R[n + 1] = surface_FSA_int_one_over_r
    flux_surf_avg_1_over_R2[n + 1] = surface_FSA_int_one_over_r2
    flux_surf_avg_grad_psi[n + 1] = surface_FSA_abs_grad_psi
    flux_surf_avg_grad_psi2[n + 1] = surface_FSA_abs_grad_psi2
    flux_surf_avg_grad_psi2_over_R2[n + 1] = surface_FSA_abs_grad_psi2_over_R2
    flux_surf_avg_B2[n + 1] = surface_FSA_B2
    flux_surf_avg_1_over_B2[n + 1] = surface_FSA_1_over_B2
    Ip[n + 1] = surface_int_bpol_dl / constants.CONSTANTS.mu_0
    delta_upper_face[n + 1] = surface_delta_upper_face
    delta_lower_face[n + 1] = surface_delta_lower_face
    elongation[n + 1] = (Z_upperextent - Z_lowerextent) / (2.0 * a_minor_local)

  # Set fsa_arrays[0] quantities
  # StandardGeometryIntermediate values at the magnetic axis are prescribed,
  # since a contour cannot be defined there.
  areas[0] = 0
  volumes[0] = 0
  R_inboard[0] = Raxis
  R_outboard[0] = Raxis
  int_dl_over_Bp[0] = 0
  flux_surf_avg_1_over_R[0] = 1 / Raxis
  flux_surf_avg_1_over_R2[0] = 1 / Raxis**2
  flux_surf_avg_grad_psi[0] = 0
  flux_surf_avg_grad_psi2[0] = 0
  flux_surf_avg_grad_psi2_over_R2[0] = 0
  flux_surf_avg_B2[0] = Btor_axis**2
  flux_surf_avg_1_over_B2[0] = 1 / Btor_axis**2
  Ip[0] = 0
  delta_upper_face[0] = delta_upper_face[1]
  delta_lower_face[0] = delta_lower_face[1]
  elongation[0] = elongation[1]

  # ------------------------------------- #
  # ---- 5. Compute derived profiles ---- #
  # ------------------------------------- #
  Phi = scipy.integrate.cumulative_trapezoid(
      q_profile, psi_on_flux_surfaces, initial=0.0
  )
  rhon = np.sqrt(Phi / Phi[-1])
  vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)

  # ------------------------------------ #
  # ---- 6. Sense-check the results ---- #
  # ------------------------------------ #
  dvolumes = np.diff(volumes)
  if not np.all(dvolumes > 0):
    idx = np.where(dvolumes <= 0)
    raise ValueError(
        'Volumes are not monotonically increasing (got decrease in volume '
        f'between surfaces {", ".join([f"{i} -> {i+1}" for i in idx[0]])}). '
        'This likely means that the contour generation failed to produce a '
        'closed flux surface at these indices. To fix, try reducing '
        'last_surface_factor or n_surfaces.'
    )

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.EQDSK,
      Ip_from_parameters=Ip_from_parameters,
      R_major=R_major,
      a_minor=a_minor,
      B_0=np.array(B_0),
      # psi_on_flux_surface has psi(0)=0 for ease of constructing the profiles
      # The absolute value of psi may have meaning when connected in wider
      # workflows, so we make sure that we set psi(0) from the eqdsk
      psi=psi_on_flux_surfaces + eq_dict['psimag'],
      Ip_profile=Ip,
      Phi=Phi,
      R_in=R_inboard,
      R_out=R_outboard,
      F=F,
      int_dl_over_Bp=int_dl_over_Bp,
      flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
      flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
      flux_surf_avg_grad_psi=flux_surf_avg_grad_psi,
      flux_surf_avg_grad_psi2=flux_surf_avg_grad_psi2,
      flux_surf_avg_grad_psi2_over_R2=flux_surf_avg_grad_psi2_over_R2,
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
      elongation=elongation,
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
      z_magnetic_axis=np.array(Zaxis),
  )


def _enforce_torax_sign_convention(
    eq_dict: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
  """Modifies the EQDSK dict to comply with TORAX conventions (COCOS11, B0>0, Ip>0)."""
  # Set Ip and B0 positive
  # This is a choice not necessarily governed by COCOS (see Table Ib in COCOS
  # paper). The logic here may need to change when rotation is added,
  # particularly combined vExB and parallel rotation, and momentum transport.
  # https://doi.org/10.1016/j.cpc.2012.09.010)
  # TODO(b/461727869): consolidate logic when full rotation features are added.
  eq_dict['cplasma'] = np.abs(eq_dict['cplasma'])
  eq_dict['bcentre'] = np.abs(eq_dict['bcentre'])
  # Ensure that psi grows with increasing radius, consistent with positive Ip
  if eq_dict['psimag'] > eq_dict['psibdry']:
    eq_dict['psi'] = -eq_dict['psi']
    eq_dict['psimag'] = -eq_dict['psimag']
    eq_dict['psibdry'] = -eq_dict['psibdry']
  # pprime and ffprime are gradients with respect to the poloidal flux (psi),
  # and its sign depends on the sign of the poloidal flux. Since pressure on
  # average should decrease as the radius increases, positive mean pprime means
  # decreasing psi which is not consistent with positive Ip, therefore we need
  # to flip the values.
  if np.mean(eq_dict['pprime']) > 0:
    eq_dict['pprime'] = -eq_dict['pprime']
    eq_dict['ffprime'] = -eq_dict['ffprime']
  return eq_dict


def _validate_torax_sign_convention(
    eq_dict: dict[str, np.ndarray | float],
) -> None:
  """Checks that an EQDSK dict complies with TORAX conventions (COCOS11, B0>0, Ip>0)."""
  # pylint: disable=invalid-name
  COCOS_violations = ''
  if eq_dict['bcentre'] < 0:
    COCOS_violations += '- B_0 is negative\n'
  if eq_dict['psibdry'] < eq_dict['psimag']:
    COCOS_violations += '- psi at the boundary is less than psi on the axis\n'
  if np.any(eq_dict['fpol']) < 0:
    COCOS_violations += '- F=RB_phi has negative values\n'
  if np.any(eq_dict['qpsi']) < 0:
    COCOS_violations += '- q has negative values\n'
  if eq_dict['cplasma'] < 0:
    COCOS_violations += '- Ip is negative'
  if COCOS_violations:
    raise ValueError(
        'The following violate the COCOS11 coordinate system when Ip and B0 are'
        ' restricted to be positive:\n'
        + COCOS_violations
        + 'Check that the COCOS of the input EQDSK was set correctly.'
    )


def _convert_eqdsk_object_to_cocos11_dict(
    cocos: int,
    eqdsk_object: eqdsk.EQDSKInterface,
) -> dict[str, np.ndarray]:
  """Converts an EQDSKInterface with arbitrary COCOS into a COCOS11 dictionary."""
  try:
    # If the EQDSK object has a COCOS set, use it.
    eqdsk_object.cocos
  except ValueError:
    # Otherwise, identify the COCOS from the object.
    eqdsk_object.identify(as_cocos=cocos)

  # Convert to TORAX COCOS.
  eqdsk_object.to_cocos(_TORAX_EQDSK_COCOS)

  return eqdsk_object.__dict__
