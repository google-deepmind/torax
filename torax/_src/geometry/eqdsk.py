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

from collections.abc import Mapping
import logging
import os
from typing import Annotated, Any, Final, Literal

import contourpy
import eqdsk
import eqdsk.file
import eqdsk.tools
from jax import numpy as jnp
import numpy as np
import numpy.typing as npt
import pydantic
import scipy
from torax._src import constants
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
from torax._src.geometry import trapped_fraction as trapped_fraction_lib
from torax._src.torax_pydantic import torax_pydantic

# Inject `npt` into eqdsk.file's runtime namespace to prevent Pydantic from
# raising `PydanticUndefinedAnnotation: name 'npt' is not defined.`
# Import of `npt` in upstream eqdsk is gated by TYPE_CHECKING, which is not
# compatible with Pydantic type validation.
# TODO(b/524519841): Upstream this fix to eqdsk.
eqdsk.file.npt = npt

# COCOS convention that TORAX translates all EQDSK geometries to.
_TORAX_EQDSK_COCOS = 11

# Below this many contour vertices, the poloidal (R, Z) equilibrium grid does
# not resolve the flux surface well enough for an accurate line integral
# (e.g. flux surfaces very close to the magnetic axis, which can be much
# smaller than a single grid cell). Surfaces below this threshold are marked
# unreliable (NaN) so the caller can smoothly extrapolate to the axis.
_MIN_CONTOUR_POINTS_FOR_EXACT_INTEGRAL: Final[int] = 20


# pylint: disable=invalid-name
class EQDSKData(torax_pydantic.ConfigBase):
  """Container for COCOS-11 normalized EQDSK equilibrium data.

  Attributes:
    nx: Number of horizontal (R) grid points.
    nz: Number of vertical (Z) grid points.
    xdim: Horizontal dimension of the computational domain [m].
    zdim: Vertical dimension of the computational domain [m].
    xcentre: Major radius of the reference vacuum field [m].
    bcentre: Reference vacuum toroidal magnetic field [T].
    xgrid1: Minimum major radius coordinate of the grid [m].
    zmid: Midplane vertical coordinate of the grid [m].
    xmag: Major radius of the magnetic axis [m].
    zmag: Vertical coordinate of the magnetic axis [m].
    psimag: Poloidal flux on the magnetic axis [Wb].
    psibdry: Poloidal flux at the plasma boundary [Wb].
    cplasma: Total plasma current [A].
    fpol: Toroidal field flux function profile F = R * B_phi [m T].
    pprime: Pressure gradient profile dp/dpsi.
    ffprime: F * dF/dpsi profile.
    qpsi: Safety factor profile q(psi).
    xbdry: Major radius coordinates of the boundary contour [m].
    zbdry: Vertical coordinates of the boundary contour [m].
    psi: 2D poloidal flux grid psi(R, Z) [Wb].
    xlim: Major radius coordinates of the limiter boundary [m] (optional).
    zlim: Vertical coordinates of the limiter boundary [m] (optional).
  """

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_eqdsk_data(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    return {k: v for k, v in data.items() if k in cls.model_fields}

  nx: Annotated[int, torax_pydantic.JAX_STATIC]
  nz: Annotated[int, torax_pydantic.JAX_STATIC]
  xdim: float
  zdim: float
  xcentre: float
  bcentre: float
  xgrid1: float
  zmid: float
  xmag: float
  zmag: float
  psimag: float
  psibdry: float
  cplasma: float
  fpol: torax_pydantic.NumpyArray1D
  pprime: torax_pydantic.NumpyArray1D
  ffprime: torax_pydantic.NumpyArray1D
  qpsi: torax_pydantic.NumpyArray1D
  xbdry: torax_pydantic.NumpyArray1D
  zbdry: torax_pydantic.NumpyArray1D
  psi: torax_pydantic.NumpyArray
  xlim: torax_pydantic.NumpyArray1D | None = None
  zlim: torax_pydantic.NumpyArray1D | None = None


class EQDSKConfig(base.GeometryConfigBase):
  """Loads an EQDSK equilibrium file or in-memory `EQDSKInterface` object.

  Traces flux surfaces from the 2D poloidal flux grid and computes the
  corresponding flux-surface averages.

  Attributes:
    model_name: Always set to `'eqdsk'`.
    eqdsk_data: COCOS-11 normalized EQDSK equilibrium data or dict across times.
    n_surfaces: Number of flux surfaces for which flux-surface averages are
      calculated.
    last_surface_factor: Multiplication factor in `(0, 1)` applied to the
      boundary poloidal flux for the contour defining geometry terms at the LCFS
      on the TORAX grid. Needed to avoid divergent integrations in diverted
      geometries.
    target_psi_grid_resolution: Desired minimum number of grid points along the
      $R$ and $Z$ dimensions of the 2D $psi$ grid before generating flux surface
      contours. Contours are linear between grid points, so for coarse EQDSK
      grids the flux surfaces near the separatrix can be inaccurate (e.g.,
      underestimating the plasma current) or fail to close. Refining the grid
      allows `last_surface_factor` to be closer to `1`. If set, the 2D $(R, Z)$
      grid is refined (using bicubic spline interpolation of $psi$) by an
      integer factor `ceil(target_psi_grid_resolution / min(nx, nz))`, refining
      both dimensions uniformly. If `None`, no refinement is performed. Must be
      `>= 1` if provided.

  Note:
    The constructor also accepts inputs normalized into eqdsk_data:
      geometry_file: File name of an EQDSK file.
      eqdsk_object: EQDSK dictionary or EQDSKInterface object.
      geometry_directory: Directory containing the EQDSK file.
      cocos: COCOS convention of the source file or object (default 11).
  """

  @property
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction_lib.TrappedFractionSource]:
    return frozenset({
        trapped_fraction_lib.TrappedFractionSource.SAUTER,
        trapped_fraction_lib.TrappedFractionSource.EXACT,
    })

  model_name: Annotated[
      Literal['eqdsk'],
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = 'eqdsk'
  eqdsk_data: EQDSKData | dict[torax_pydantic.Second, EQDSKData]
  n_surfaces: Annotated[pydantic.PositiveInt, torax_pydantic.JAX_STATIC] = 100
  last_surface_factor: Annotated[
      float, pydantic.Field(gt=0.0, le=1.0), torax_pydantic.JAX_STATIC
  ] = 0.99
  target_psi_grid_resolution: Annotated[
      int | None, pydantic.Field(ge=1), torax_pydantic.JAX_STATIC
  ] = 128

  @pydantic.model_validator(mode='before')
  @classmethod
  def _before_validator(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    data_dict = dict(data)
    if 'eqdsk_data' in data_dict:
      data_dict.pop('geometry_file', None)
      data_dict.pop('eqdsk_object', None)
      data_dict.pop('geometry_directory', None)
      data_dict.pop('cocos', None)
      return data_dict

    geometry_file = data_dict.pop('geometry_file', None)
    eqdsk_object = data_dict.pop('eqdsk_object', None)
    geometry_directory = data_dict.pop('geometry_directory', None)
    cocos = data_dict.pop('cocos', None)

    if geometry_file is not None and eqdsk_object is not None:
      raise ValueError(
          "Cannot provide both 'geometry_file' and 'eqdsk_object'."
      )

    if (
        geometry_file is None
        and eqdsk_object is None
        and 'eqdsk_data' not in data_dict
    ):
      raise ValueError(
          "Either 'geometry_file' or 'eqdsk_object' must be provided."
      )

    if cocos is None:
      cocos = 11

    def _process_one_eqdsk(file_or_obj: Any) -> Any:
      if isinstance(file_or_obj, (str, os.PathLike)):
        res = geometry_loader.load_geo_data(
            geometry_dir=geometry_directory,
            geometry_file=file_or_obj,
            geometry_source=geometry_loader.GeometrySource.EQDSK,
            cocos=cocos,
        )
      else:
        obj = file_or_obj
        if isinstance(obj, dict) and 'xbdry' not in obj:
          obj = eqdsk.EQDSKInterface.from_dict(obj, no_cocos=True)
        res = _convert_eqdsk_object_to_cocos11_dict(cocos, obj)
      res = _enforce_torax_sign_convention(res)
      _validate_torax_sign_convention(res)
      return res

    if isinstance(geometry_file, Mapping):
      data_dict['eqdsk_data'] = {
          float(t): _process_one_eqdsk(f) for t, f in geometry_file.items()
      }
    elif (
        isinstance(eqdsk_object, Mapping)
        and not hasattr(eqdsk_object, 'xbdry')
        and not (isinstance(eqdsk_object, dict) and 'xbdry' in eqdsk_object)
    ):
      data_dict['eqdsk_data'] = {
          float(t): _process_one_eqdsk(obj) for t, obj in eqdsk_object.items()
      }
    elif eqdsk_object is not None:
      data_dict['eqdsk_data'] = _process_one_eqdsk(eqdsk_object)
    else:
      assert geometry_file is not None
      data_dict['eqdsk_data'] = _process_one_eqdsk(geometry_file)

    return data_dict

  def _build_slice_geometry(
      self, data: EQDSKData
  ) -> standard_geometry.StandardGeometry:
    intermediates = _construct_intermediates_from_eqdsk(
        eqdsk_data=data,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
        n_surfaces=self.n_surfaces,
        last_surface_factor=self.last_surface_factor,
        target_psi_grid_resolution=self.target_psi_grid_resolution,
        trapped_fraction_source=self.trapped_fraction_source,
    )
    return standard_geometry.build_standard_geometry(intermediates)

  def _build_geometry(
      self,
  ) -> (
      standard_geometry.StandardGeometry
      | Mapping[float, standard_geometry.StandardGeometry]
  ):
    if isinstance(self.eqdsk_data, EQDSKData):
      return self._build_slice_geometry(self.eqdsk_data)
    return {
        float(t): self._build_slice_geometry(data)
        for t, data in self.eqdsk_data.items()
    }


def _is_closed(contour: np.ndarray) -> bool:
  """Returns whether a `contourpy` contour line is a closed loop.

  `contourpy` marks a closed loop by repeating its first vertex at the end, so
  a non-degenerate closed polygon (which requires at least three distinct
  vertices to enclose an area rather than collapsing to a point or line segment)
  has length >= 4. Lines that run into the edge of the contouring domain are
  open and have different first and last vertices.

  Args:
    contour: (N, 2) array of (R, Z) vertices as returned by `contourpy`.

  Returns:
    True if `contour` is a closed loop with at least three distinct vertices.
  """
  return len(contour) >= 4 and np.array_equal(contour[0], contour[-1])


def _encloses_point(polygon: np.ndarray, r: float, z: float) -> bool:
  """Returns whether a polygon encloses the point (r, z).

  Uses the standard ray-casting (even-odd) test: cast a horizontal ray from
  (r, z) and count how many polygon edges it crosses. The point is inside if and
  only if the count is odd.

  Args:
    polygon: (N, 2) array of (R, Z) vertices. The polygon is implicitly closed
      (the last vertex connects back to the first), so the first vertex must not
      be repeated at the end.
    r: R coordinate of the point.
    z: Z coordinate of the point.

  Returns:
    True if `(r, z)` lies strictly inside `polygon`.
  """
  # Edge i runs from vertex i, (r_start, z_start), to vertex i + 1,
  # (r_end, z_end), wrapping around from the last vertex to the first.
  r_start, z_start = polygon[:, 0], polygon[:, 1]
  r_end, z_end = np.roll(r_start, -1), np.roll(z_start, -1)

  # An edge can only cross the ray if its endpoints lie on opposite sides of
  # the horizontal line through z. The half-open comparison (> z) counts a
  # vertex lying exactly at height z on one side only, so it is never counted
  # twice.
  straddles_ray = (z_start > z) != (z_end > z)

  # R coordinate where each straddling edge crosses the horizontal line through
  # z. Non-straddling edges (including horizontal edges with z_end == z_start)
  # are replaced by a unit denominator and masked out below.
  dz = np.where(straddles_ray, z_end - z_start, 1.0)
  r_crossing = r_start + (r_end - r_start) * (z - z_start) / dz

  # The ray extends to the right, so only crossings to the right of r count.
  crosses_ray = straddles_ray & (r < r_crossing)
  return bool(np.sum(crosses_ray) % 2 == 1)


def _select_contour_enclosing_magnetic_axis(
    contours: list[np.ndarray], r_axis: float, z_axis: float
) -> np.ndarray | None:
  """Selects the closed contour that encloses the magnetic axis.

  Contours are only computed inside the bounding box of the LCFS, not inside
  the LCFS itself. A single psi level can therefore produce several contour
  lines: the closed flux surface around the magnetic axis, plus pieces outside
  the plasma that lie inside the box (e.g. in the private-flux region below
  the X-point). The returned flux surface is the closed contour enclosing the
  magnetic axis.

  Args:
    contours: Contour lines from `contourpy` at a single psi level, each an (N,
      2) array of (R, Z) vertices. Closed lines repeat their first vertex at the
      end.
    r_axis: R coordinate of the magnetic axis.
    z_axis: Z coordinate of the magnetic axis.

  Returns:
    (N, 2) array of (R, Z) vertices of the closed flux surface with its
    repeated closing vertex removed, or None if no closed contour encloses the
    magnetic axis.

  Raises:
    ValueError: If more than one closed contour encloses the magnetic axis.
  """
  enclosing_contours = [
      contour[:-1]
      for contour in contours
      if _is_closed(contour) and _encloses_point(contour[:-1], r_axis, z_axis)
  ]
  if len(enclosing_contours) > 1:
    raise ValueError(
        'Expected at most one closed contour enclosing the magnetic axis'
        f' (R={r_axis}, Z={z_axis}), but found {len(enclosing_contours)}.'
    )
  return enclosing_contours[0] if enclosing_contours else None


def _line_elements(r: np.ndarray, z: np.ndarray) -> np.ndarray:
  """Returns the line element dl at each vertex of a closed curve.

  Flux-surface integrals are evaluated as sums over the curve vertices,
  `integral(f dl) ~= sum_i f_i * dl_i`. On each straight segment of length `L_i`
  between vertex `i` and vertex `i + 1` (with the last vertex connected back to
  the first), the trapezoidal rule approximates the segment integral as
  `0.5 * L_i * (f_i + f_{i+1})`. Summing over all segments collects a weight
  `dl_i = 0.5 * (L_{i-1} + L_i)` at each vertex `i` (half the length of the
  segment before it plus half the length of the segment after it). The result
  does not depend on which vertex the curve starts at, and `sum_i dl_i` equals
  the polygon perimeter.

  Args:
    r: R coordinates of the closed curve vertices, without repeating the first
      vertex at the end.
    z: Z coordinates of the closed curve vertices, with the same convention as
      `r`.

  Returns:
    Array of the same length as `r` with the line element at each vertex.
  """
  # segment_lengths[i] is the distance from vertex i to vertex i + 1,
  # wrapping around from the last vertex to the first.
  segment_lengths = np.hypot(np.roll(r, -1) - r, np.roll(z, -1) - z)
  # Vertex i is shared by segments i - 1 and i.
  return 0.5 * (np.roll(segment_lengths, 1) + segment_lengths)


def _construct_intermediates_from_eqdsk(
    *,
    eqdsk_data: EQDSKData,
    face_centers: np.ndarray,
    hires_factor: int,
    n_surfaces: int,
    last_surface_factor: float,
    target_psi_grid_resolution: int | None,
    trapped_fraction_source: (
        trapped_fraction_lib.TrappedFractionSource
    ) = trapped_fraction_lib.TrappedFractionSource.SAUTER,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from EQDSK.

  This method constructs a StandardGeometryIntermediates object from an EQDSK
  file or object. It calculates flux surface averages based on the EQDSK
  geometry 2D psi mesh.

  Args:
    eqdsk_data: EQDSKData or dictionary in COCOS-11.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    n_surfaces: Number of surfaces for which flux surface averages are
      calculated.
    last_surface_factor: Multiplication factor of the boundary poloidal flux,
      used for the contour defining geometry terms at the LCFS on the TORAX
      grid. Needed to avoid divergent integrations in diverted geometries.
    target_psi_grid_resolution: Desired minimum number of grid points along the
      R and Z dimensions of the 2D psi grid before generating flux surface
      contours. If set, the grid is refined by `ceil(target / min_dim)`, where
      `target = target_psi_grid_resolution` and `min_dim = min(nx, nz)`,
      refining both dimensions uniformly. If None, no refinement is performed.
    trapped_fraction_source: Selects how the effective trapped particle fraction
      is computed; see `trapped_fraction.TrappedFractionSource`.

  Returns:
    A StandardGeometryIntermediates instance based on the input data.
    This can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """

  def calculate_area(x, z):
    """Gauss-shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)."""
    z_shifted = np.roll(z, -1)
    x_shifted = np.roll(x, -1)
    return 0.5 * np.abs(np.sum(x * z_shifted - z * x_shifted))

  # Reference geometry terms
  # TODO(b/375696414): deal with updown asymmetric cases.
  # R_major taken as Rgeo (LCFS R_major)
  R_major = (eqdsk_data.xbdry.max() + eqdsk_data.xbdry.min()) / 2.0
  a_minor = (eqdsk_data.xbdry.max() - eqdsk_data.xbdry.min()) / 2.0
  # Vacuum toroidal magnetic field at R_major. EQDSK vacuum field, bcentre, is
  # defined at an arbitrary reference radius, xcentre. To get B at R_major,
  # use B_vacuum ∝ 1/R.

  # See EQDSKInterface in https://github.com/Fusion-Power-Plant-Framework/eqdsk
  B_0 = eqdsk_data.bcentre * eqdsk_data.xcentre / R_major
  Raxis = float(eqdsk_data.xmag)
  Zaxis = float(eqdsk_data.zmag)
  Btor_axis = eqdsk_data.fpol[0] / eqdsk_data.xmag

  # 1D psi grid, with psi(axis) = 0
  psi_1dgrid = np.linspace(
      0.0, eqdsk_data.psibdry - eqdsk_data.psimag, eqdsk_data.nx
  )

  # 2D psi grid, with psi(axis) = 0
  psi_2dgrid = eqdsk_data.psi - eqdsk_data.psimag

  # Spline interpolator of 2D psi field defined on raw EQDSK X-Z grid
  raw_X_1D = np.linspace(
      eqdsk_data.xgrid1, eqdsk_data.xgrid1 + eqdsk_data.xdim, eqdsk_data.nx
  )
  raw_Z_1D = np.linspace(
      eqdsk_data.zmid - eqdsk_data.zdim / 2,
      eqdsk_data.zmid + eqdsk_data.zdim / 2,
      eqdsk_data.nz,
  )
  psi_2dgrid_interpolator = scipy.interpolate.RectBivariateSpline(
      raw_X_1D, raw_Z_1D, psi_2dgrid, kx=3, ky=3, s=0
  )

  # Optionally refine the 2D psi grid.
  if target_psi_grid_resolution is not None:
    # Convert target resolution to refinement factor.
    min_resolution = min(eqdsk_data.nx, eqdsk_data.nz)
    refinement_factor = int(
        jnp.ceil(target_psi_grid_resolution / min_resolution)
    )
  else:
    # No refinement.
    refinement_factor = 1

  # 2D X-Z grid
  if refinement_factor > 1:
    nx = (eqdsk_data.nx - 1) * refinement_factor + 1
    nz = (eqdsk_data.nz - 1) * refinement_factor + 1
    X_1D = np.linspace(raw_X_1D[0], raw_X_1D[-1], nx)
    Z_1D = np.linspace(raw_Z_1D[0], raw_Z_1D[-1], nz)
    psi_2dgrid = psi_2dgrid_interpolator(X_1D, Z_1D)
  else:
    X_1D = raw_X_1D
    Z_1D = raw_Z_1D

  X, Z = np.meshgrid(X_1D, Z_1D, indexing='ij')
  Xlcfs, Zlcfs = eqdsk_data.xbdry, eqdsk_data.zbdry

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
      (eqdsk_data.psibdry - eqdsk_data.psimag) * last_surface_factor,
      n_surfaces,
  )

  # Each entry of `surfaces` is (R vertices, Z vertices) for a closed flux
  # surface around the magnetic axis, without a repeated closing vertex.
  surfaces = []
  # psi values for which no closed contour around the magnetic axis was found.
  psi_without_closed_surface = []
  psi_contour_generator = contourpy.contour_generator(X, Z, masked_psi_2dgrid)

  # Skip magnetic axis since no contour is defined there.
  for _, _psi in enumerate(psi_on_flux_surfaces[1:]):
    contours = psi_contour_generator.create_contour(_psi)
    if not contours:
      raise ValueError(f"""
          Valid contour not found for EQDSK geometry for psi value {_psi}.
          Possible reason is too many surfaces requested.
          Try reducing n_surfaces from the current value of {n_surfaces}.
          """)
    contour = _select_contour_enclosing_magnetic_axis(contours, Raxis, Zaxis)
    if contour is None:
      psi_without_closed_surface.append(_psi)
      continue
    surfaces.append((contour[:, 0], contour[:, 1]))
  if psi_without_closed_surface:
    raise ValueError(
        'No closed contour enclosing the magnetic axis found for'
        f' {len(psi_without_closed_surface)} EQDSK flux surface(s), psi in'
        f' [{min(psi_without_closed_surface)},'
        f' {max(psi_without_closed_surface)}]. Consider increasing'
        ' target_psi_grid_resolution, or reducing last_surface_factor or'
        ' n_surfaces.'
    )

  # ------------------------------------------------------------------ #
  # ---- 3. Interpolate everything onto the new flux surface grid ---- #
  # ------------------------------------------------------------------ #
  # Interpolate safety factor onto new flux-surface grid
  q_interpolator = scipy.interpolate.interp1d(
      psi_1dgrid, eqdsk_data.qpsi, kind='cubic'
  )
  q_profile = q_interpolator(psi_on_flux_surfaces)

  # Interpolate toroidal field flux function onto new flux-surface grid
  F_interpolator = scipy.interpolate.interp1d(
      psi_1dgrid, eqdsk_data.fpol, kind='cubic'
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
  trapped_fraction = np.full(len(surfaces) + 1, np.nan)
  int_dl_over_Bp = np.empty(len(surfaces) + 1)  # int(Rdl / | grad(psi) |)
  I_p = np.empty(len(surfaces) + 1)  # Toroidal plasma current
  delta_upper_face = np.empty(len(surfaces) + 1)  # Upper face delta
  delta_lower_face = np.empty(len(surfaces) + 1)  # Lower face delta
  elongation = np.empty(len(surfaces) + 1)  # Elongation

  # Compute fsa for each surface
  # Note: surfaces is from psi[1:]
  for n, (x_surface, z_surface) in enumerate(surfaces):
    surface_dl = _line_elements(x_surface, z_surface)

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
    if (
        trapped_fraction_source
        == trapped_fraction_lib.TrappedFractionSource.EXACT
    ):
      if len(x_surface) < _MIN_CONTOUR_POINTS_FOR_EXACT_INTEGRAL:
        trapped_fraction[n + 1] = np.nan
      else:
        trapped_fraction[n + 1] = (
            trapped_fraction_lib.calculate_bounce_averaged_trapped_fraction(
                B=np.sqrt(surface_B2),
                # Same line elements as the flux-surface averages above.
                dl_over_Bp=surface_dl / surface_Bpol,
            )
        )
    I_p[n + 1] = surface_int_bpol_dl / constants.CONSTANTS.mu_0
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
  if (
      trapped_fraction_source
      == trapped_fraction_lib.TrappedFractionSource.EXACT
  ):
    # No trapped particles on the magnetic axis, where B is uniform.
    trapped_fraction[0] = 0.0
  I_p[0] = 0
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

  epsilon = (R_outboard - R_inboard) / (R_outboard + R_inboard)
  delta = (delta_upper_face + delta_lower_face) / 2.0
  sauter_trapped_fraction = (
      trapped_fraction_lib.calculate_sauter_trapped_fraction(
          epsilon=epsilon, delta=delta
      )
  )

  match trapped_fraction_source:
    case trapped_fraction_lib.TrappedFractionSource.EXACT:
      # Fill any unreliable values (NaN, or outside the physically valid
      # [0, 1] range, e.g. surfaces too close to the magnetic axis for the
      # integral to resolve well) via smooth extrapolation to the axis.
      exact_is_unreliable = (
          np.isnan(trapped_fraction)
          | (trapped_fraction < 0.0)
          | (trapped_fraction > 1.0)
      )
      if np.any(exact_is_unreliable):
        logging.warning(
            'Overwriting %d bad/unphysical EXACT trapped fraction values via'
            ' smooth extrapolation to the axis.',
            int(np.sum(exact_is_unreliable)),
        )
      trapped_fraction = (
          trapped_fraction_lib.extrapolate_exact_trapped_fraction_to_axis(
              trapped_fraction=trapped_fraction,
              rhon=rhon,
              sauter_trapped_fraction=sauter_trapped_fraction,
          )
      )
    case trapped_fraction_lib.TrappedFractionSource.SAUTER:
      trapped_fraction = sauter_trapped_fraction
    case _:
      raise ValueError(
          f'Unsupported trapped_fraction_source: {trapped_fraction_source}.'
      )

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
      R_major=R_major,
      a_minor=a_minor,
      B_0=np.array(B_0),
      # psi_on_flux_surface has psi(0)=0 for ease of constructing the profiles
      # The absolute value of psi may have meaning when connected in wider
      # workflows, so we make sure that we set psi(0) from the eqdsk
      psi=psi_on_flux_surfaces + eqdsk_data.psimag,
      I_p_profile=I_p,
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
      trapped_fraction=trapped_fraction,
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
  """Modifies the EQDSK dict to comply with TORAX conventions (COCOS11, B_0>0, I_p>0)."""
  # Set I_p and B_0 positive
  # This is a choice not necessarily governed by COCOS (see Table Ib in COCOS
  # paper). The logic here may need to change when rotation is added,
  # particularly combined vExB and parallel rotation, and momentum transport.
  # https://doi.org/10.1016/j.cpc.2012.09.010)
  # TODO(b/461727869): consolidate logic when full rotation features are added.
  eq_dict['cplasma'] = np.abs(eq_dict['cplasma'])
  eq_dict['bcentre'] = np.abs(eq_dict['bcentre'])
  # Ensure that psi grows with increasing radius, consistent with positive I_p
  if eq_dict['psimag'] > eq_dict['psibdry']:
    eq_dict['psi'] = -eq_dict['psi']
    eq_dict['psimag'] = -eq_dict['psimag']
    eq_dict['psibdry'] = -eq_dict['psibdry']
  # pprime and ffprime are gradients with respect to the poloidal flux (psi),
  # and its sign depends on the sign of the poloidal flux. Since pressure on
  # average should decrease as the radius increases, positive mean pprime means
  # decreasing psi which is not consistent with positive I_p, therefore we need
  # to flip the values.
  if np.mean(eq_dict['pprime']) > 0:
    eq_dict['pprime'] = -eq_dict['pprime']
    eq_dict['ffprime'] = -eq_dict['ffprime']
  return eq_dict


def _validate_torax_sign_convention(
    eq_dict: Mapping[str, Any],
) -> None:
  """Checks that an EQDSK dict complies with TORAX conventions (COCOS11, B_0>0, I_p>0)."""
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
    COCOS_violations += '- I_p is negative'
  if COCOS_violations:
    raise ValueError(
        'The following violate the COCOS11 coordinate system when I_p and B_0'
        ' are restricted to be positive:\n'
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
