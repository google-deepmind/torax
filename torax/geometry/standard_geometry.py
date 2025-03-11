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

"""Classes for representing a standard geometry.

This is a geometry object that is used for most geometries sources
CHEASE, FBT, etc.
"""


from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools

import chex
import contourpy
import numpy as np
import scipy
from torax import constants
from torax import interpolated_param
from torax import jax_utils
from torax.geometry import geometry
from torax.geometry import geometry_loader
from torax.geometry import geometry_provider
from torax.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

_RHO_SMOOTHING_LIMIT = 0.1


@chex.dataclass(frozen=True)
class StandardGeometry(geometry.Geometry):
  r"""Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.  This class extends
  the base `Geometry` class with attributes that are commonly computed
  from various equilibrium data sources (CHEASE, FBT, EQDSK, etc.).

  Attributes:
    Ip_from_parameters: Boolean indicating whether the total plasma current
      (`Ip_tot`) is determined by the config parameters (True) or read from the
      geometry file (False).
    Ip_profile_face: Plasma current profile on the face grid
      [:math:`\mathrm{A}`].
    psi: 1D poloidal flux profile on the cell grid [:math:`\mathrm{Wb}`].
    psi_from_Ip: Poloidal flux profile on the cell grid  [:math:`\mathrm{Wb}`],
      calculated from the plasma current profile in the geometry file.
    psi_from_Ip_face: Poloidal flux profile on the face grid [Wb], calculated
      from the plasma current profile in the geometry file.
    jtot: Total toroidal current density profile on the cell grid
      [:math:`\mathrm{A/m^2}`].
    jtot_face: Total toroidal current density profile on the face grid
      [:math:`\mathrm{A/m^2}`].
    delta_upper_face: Upper triangularity on the face grid [dimensionless]. See
      `Geometry` docstring for definition of `delta_upper_face`.
    delta_lower_face: Lower triangularity on the face grid [dimensionless]. See
      `Geometry` docstring for definition of `delta_lower_face`.
  """

  Ip_from_parameters: bool
  Ip_profile_face: chex.Array
  psi: chex.Array
  psi_from_Ip: chex.Array
  psi_from_Ip_face: chex.Array
  jtot: chex.Array
  jtot_face: chex.Array
  delta_upper_face: chex.Array
  delta_lower_face: chex.Array


@chex.dataclass(frozen=True)
class StandardGeometryProvider(geometry_provider.TimeDependentGeometryProvider):
  """Values to be interpolated for a Standard Geometry."""

  Ip_from_parameters: bool
  Ip_profile_face: interpolated_param.InterpolatedVarSingleAxis
  psi: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip_face: interpolated_param.InterpolatedVarSingleAxis
  jtot: interpolated_param.InterpolatedVarSingleAxis
  jtot_face: interpolated_param.InterpolatedVarSingleAxis
  delta_upper_face: interpolated_param.InterpolatedVarSingleAxis
  delta_lower_face: interpolated_param.InterpolatedVarSingleAxis
  elongation: interpolated_param.InterpolatedVarSingleAxis
  elongation_face: interpolated_param.InterpolatedVarSingleAxis

  @functools.partial(jax_utils.jit, static_argnums=0)
  def __call__(self, t: chex.Numeric) -> geometry.Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, StandardGeometry)


@dataclasses.dataclass(frozen=True)
class StandardGeometryIntermediates:
  r"""Holds the intermediate values used to build a StandardGeometry.

  In particular these are the values that are used when interpolating different
  geometries.  These intermediates are typically extracted directly from
  equilibrium solver outputs (like CHEASE, FBT, or EQDSK) and then used to
  construct a `StandardGeometry` instance.

  TODO(b/335204606): Specify the expected COCOS format.
  NOTE: Right now, TORAX does not have a specified COCOS format. Our team is
  working on adding this and updating documentation to make that clear. The
  CHEASE input data is still COCOS 2.

  All inputs are 1D profiles vs normalized rho toroidal (rhon).

  Attributes:
    geometry_type:  The type of geometry being represented (e.g., CHEASE, FBT,
      EQDSK).
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    Rmaj: major radius on the magnetic axis in [:math:`\mathrm{m}`].
    Rmin: minor radius (a) in [:math:`\mathrm{m}`].
    B: Toroidal magnetic field on axis [:math:`\mathrm{T}`].
    psi: Poloidal flux profile [:math:`\mathrm{Wb}`].
    Ip_profile: Plasma current profile [:math:`\mathrm{A}`].
    Phi: Toroidal flux profile [:math:`\mathrm{Wb}`].
    Rin: Radius of the flux surface at the inboard side at midplane
      [:math:`\mathrm{m}`]. Inboard side is defined as the innermost radius.
    Rout: Radius of the flux surface at the outboard side at midplane
      [:math:`\mathrm{m}`]. Outboard side is defined as the outermost radius.
    F: Toroidal field flux function (:math:`F = R B_{\phi}`) [:math:`\mathrm{m
      T}`].
    int_dl_over_Bp: :math:`\oint dl/B_p` (field-line contour integral on the
      flux surface) [:math:`\mathrm{m / T}`], where :math:`B_p` is the poloidal
      magnetic field.
    flux_surf_avg_1_over_R2: Flux surface average of :math:`1/R^2`
      [:math:`\mathrm{m^{-2}}`].
    flux_surf_avg_Bp2: Flux surface average of :math:`B_p^2`
      [:math:`\mathrm{T^2}`].
    flux_surf_avg_RBp: Flux surface average of :math:`R B_p` [:math:`\mathrm{m
      T}`].
    flux_surf_avg_R2Bp2: Flux surface average of :math:`R^2 B_p^2`
      [:math:`\mathrm{m^2 T^2}`].
    delta_upper_face: Upper triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    delta_lower_face: Lower triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    elongation: Plasma elongation profile [dimensionless]. See `Geometry`
      docstring for definition.
    vpr:  Profile of dVolume/d(rho_norm), where rho_norm is the normalized
      toroidal flux coordinate [:math:`\mathrm{m^3}`].
    n_rho: Radial grid points (number of cells).
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations. Used to create a higher-resolution grid to improve accuracy
      when initializing psi from a plasma current profile.
    z_magnetic_axis: z position of magnetic axis [:math:`\mathrm{m}`].
  """

  geometry_type: geometry.GeometryType
  Ip_from_parameters: bool
  Rmaj: chex.Numeric
  Rmin: chex.Numeric
  B: chex.Numeric
  psi: chex.Array
  Ip_profile: chex.Array
  Phi: chex.Array
  Rin: chex.Array
  Rout: chex.Array
  F: chex.Array
  int_dl_over_Bp: chex.Array
  flux_surf_avg_1_over_R2: chex.Array
  flux_surf_avg_Bp2: chex.Array
  flux_surf_avg_RBp: chex.Array
  flux_surf_avg_R2Bp2: chex.Array
  delta_upper_face: chex.Array
  delta_lower_face: chex.Array
  elongation: chex.Array
  vpr: chex.Array
  n_rho: int
  hires_fac: int
  z_magnetic_axis: chex.Numeric | None

  def __post_init__(self):
    """Extrapolates edge values and smooths near-axis values.

    - Edge extrapolation for a subset of attributes based on a Cubic spline fit.
    - Near-axis smoothing for a subset of attributes based on a Savitzky-Golay
      filter with an appropriate polynominal order based on the attribute.
    """

    # Check if last flux surface is diverted and correct via spline fit if so
    if self.flux_surf_avg_Bp2[-1] < 1e-10:
      # Calculate rhon
      rhon = np.sqrt(self.Phi / self.Phi[-1])

      # Create a lambda function for the Cubic spline fit.
      spline = lambda rho, data, x, bc_type: scipy.interpolate.CubicSpline(
          rho[:-1],
          data[:-1],
          bc_type=bc_type,
      )(x)

      # Decide on the bc_type based on demanding monotonic behaviour of g2.
      # Natural bc_type means no second derivative at the spline edge, and will
      # maintain monotonicity on extrapolation, but not recommended as default.
      flux_surf_avg_Bp2_edge = spline(
          rhon,
          self.flux_surf_avg_Bp2,
          1.0,
          bc_type='not-a-knot',
      )
      int_dl_over_Bp_edge = spline(
          rhon,
          self.int_dl_over_Bp,
          1.0,
          bc_type='not-a-knot',
      )
      g2_edge_ratio = (flux_surf_avg_Bp2_edge * int_dl_over_Bp_edge**2) / (
          self.flux_surf_avg_Bp2[-2] * self.int_dl_over_Bp[-2] ** 2
      )
      if g2_edge_ratio > 1.0:
        bc_type = 'not-a-knot'
      else:
        bc_type = 'natural'
      set_edge = lambda array: spline(rhon, array, 1.0, bc_type)
      self.int_dl_over_Bp[-1] = set_edge(self.int_dl_over_Bp)
      self.flux_surf_avg_Bp2[-1] = set_edge(self.flux_surf_avg_Bp2)
      self.flux_surf_avg_1_over_R2[-1] = set_edge(self.flux_surf_avg_1_over_R2)
      self.flux_surf_avg_RBp[-1] = set_edge(self.flux_surf_avg_RBp)
      self.flux_surf_avg_R2Bp2[-1] = set_edge(self.flux_surf_avg_R2Bp2)
      self.vpr[-1] = set_edge(self.vpr)

    # Near-axis smoothing of quantities with known near-axis trends with rho
    rhon = np.sqrt(self.Phi / self.Phi[-1])
    idx_limit = np.argmin(np.abs(rhon - _RHO_SMOOTHING_LIMIT))

    # Bp goes like rho near-axis. So Bp2 terms are smoothed with order 2,
    # and Bp terms with order 1. vpr also goes like rho near-axis
    self.flux_surf_avg_Bp2[:] = _smooth_savgol(
        self.flux_surf_avg_Bp2, idx_limit, 2
    )
    self.flux_surf_avg_R2Bp2[:] = _smooth_savgol(
        self.flux_surf_avg_R2Bp2, idx_limit, 2
    )
    self.flux_surf_avg_RBp[:] = _smooth_savgol(
        self.flux_surf_avg_RBp, idx_limit, 1
    )
    self.vpr[:] = _smooth_savgol(self.vpr, idx_limit, 1)

  @classmethod
  def from_chease(
      cls,
      geometry_dir: str | None,
      geometry_file: str,
      Ip_from_parameters: bool,
      n_rho: int,
      Rmaj: float,
      Rmin: float,
      B0: float,
      hires_fac: int,
  ) -> StandardGeometryIntermediates:
    """Constructs a StandardGeometryIntermediates from a CHEASE file.

    Args:
      geometry_dir: Directory where to find the CHEASE file describing the
        magnetic geometry. If None, uses the environment variable
        TORAX_GEOMETRY_DIR if available. If that variable is not set and
        geometry_dir is not provided, then it defaults to another dir. See
        implementation.
      geometry_file: CHEASE file name.
      Ip_from_parameters: If True, the Ip is taken from the parameters and the
        values in the Geometry are rescaled to match the new Ip.
      n_rho: Radial grid points (num cells)
      Rmaj: major radius (R) in meters. CHEASE geometries are normalized, so
        this is used as an unnormalization factor.
      Rmin: minor radius (a) in meters
      B0: Toroidal magnetic field on axis [T].
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations.

    Returns:
      A StandardGeometry instance based on the input file. This can then be
      used to build a StandardGeometry by passing to `build_standard_geometry`.
    """
    chease_data = geometry_loader.load_geo_data(
        geometry_dir, geometry_file, geometry_loader.GeometrySource.CHEASE
    )

    # Prepare variables from CHEASE to be interpolated into our simulation
    # grid. CHEASE variables are normalized. Need to unnormalize them with
    # reference values poloidal flux and CHEASE-internal-calculated plasma
    # current.
    psiunnormfactor = Rmaj**2 * B0

    # set psi in TORAX units with 2*pi factor
    psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor * 2 * np.pi
    Ip_chease = chease_data['Ipprofile'] / constants.CONSTANTS.mu0 * Rmaj * B0

    # toroidal flux
    Phi = (chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] * Rmaj) ** 2 * B0 * np.pi

    # midplane radii
    Rin_chease = chease_data['R_INBOARD'] * Rmaj
    Rout_chease = chease_data['R_OUTBOARD'] * Rmaj
    # toroidal field flux function
    F = chease_data['T=RBphi'] * Rmaj * B0

    int_dl_over_Bp = chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] * Rmaj / B0
    flux_surf_avg_1_over_R2 = chease_data['<1/R**2>'] / Rmaj**2
    flux_surf_avg_Bp2 = chease_data['<Bp**2>'] * B0**2
    flux_surf_avg_RBp = chease_data['<|grad(psi)|>'] * psiunnormfactor / Rmaj
    flux_surf_avg_R2Bp2 = (
        chease_data['<|grad(psi)|**2>'] * psiunnormfactor**2 / Rmaj**2
    )

    rhon = np.sqrt(Phi / Phi[-1])
    vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)

    return cls(
        geometry_type=geometry.GeometryType.CHEASE,
        Ip_from_parameters=Ip_from_parameters,
        Rmaj=np.array(Rmaj),
        Rmin=np.array(Rmin),
        B=np.array(B0),
        psi=psi,
        Ip_profile=Ip_chease,
        Phi=Phi,
        Rin=Rin_chease,
        Rout=Rout_chease,
        F=F,
        int_dl_over_Bp=int_dl_over_Bp,
        flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
        flux_surf_avg_Bp2=flux_surf_avg_Bp2,
        flux_surf_avg_RBp=flux_surf_avg_RBp,
        flux_surf_avg_R2Bp2=flux_surf_avg_R2Bp2,
        delta_upper_face=chease_data['delta_upper'],
        delta_lower_face=chease_data['delta_bottom'],
        elongation=chease_data['elongation'],
        vpr=vpr,
        n_rho=n_rho,
        hires_fac=hires_fac,
        z_magnetic_axis=None,
    )

  @classmethod
  def from_fbt_single_slice(
      cls,
      geometry_dir: str | None,
      LY_object: str | Mapping[str, np.ndarray],
      L_object: str | Mapping[str, np.ndarray],
      Ip_from_parameters: bool = True,
      n_rho: int = 25,
      hires_fac: int = 4,
  ) -> StandardGeometryIntermediates:
    """Returns StandardGeometryIntermediates from a single slice FBT LY file.

    LY and L are FBT data files containing magnetic geometry information.
    The majority of the needed information is in the LY file. The L file
    is only needed to get the normalized poloidal flux coordinate, pQ.

    This method is for cases when the LY file on disk corresponds to a single
    time slice. Either a single time slice or sequence of time slices can be
    provided in the geometry config.

    Args:
      geometry_dir: Directory where to find the FBT file describing the magnetic
        geometry. If None, uses the environment variable TORAX_GEOMETRY_DIR if
        available. If that variable is not set and geometry_dir is not provided,
        then it defaults to another dir. See `load_geo_data` implementation.
      LY_object: File name for LY data, or directly an LY single slice dict.
      L_object: File name for L data, or directly an L dict.
      Ip_from_parameters: If True, then Ip is taken from the config and the
        values in the Geometry are rescaled
      n_rho: Grid resolution used for all TORAX cell variables.
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations.

    Returns:
      A StandardGeometryIntermediates instance based on the input slice. This
      can then be used to build a StandardGeometry by passing to
      `build_standard_geometry`.
    """
    if isinstance(LY_object, str):
      LY = geometry_loader.load_geo_data(
          geometry_dir, LY_object, geometry_loader.GeometrySource.FBT
      )
    elif isinstance(LY_object, Mapping):
      LY = LY_object
    else:
      raise ValueError(
          'LY_object must be a string (file path) or a dictionary.'
      )
    if isinstance(L_object, str):
      L = geometry_loader.load_geo_data(
          geometry_dir, L_object, geometry_loader.GeometrySource.FBT
      )
    elif isinstance(L_object, Mapping):
      L = L_object
    else:
      raise ValueError('L_object must be a string (file path) or a dictionary.')

    # Convert any scalar LY values to ndarrays such that validation method works
    for key in LY:
      if not isinstance(LY[key], np.ndarray):
        LY[key] = np.array(LY[key])

    # Raises a ValueError if the data is invalid.
    _validate_fbt_data(LY, L)
    return cls._from_fbt(LY, L, Ip_from_parameters, n_rho, hires_fac)

  @classmethod
  def from_fbt_bundle(
      cls,
      geometry_dir: str | None,
      LY_bundle_object: str | Mapping[str, np.ndarray],
      L_object: str | Mapping[str, np.ndarray],
      LY_to_torax_times: np.ndarray | None,
      Ip_from_parameters: bool = True,
      n_rho: int = 25,
      hires_fac: int = 4,
  ) -> Mapping[float, StandardGeometryIntermediates]:
    """Returns StandardGeometryIntermediates from a bundled FBT LY file.

    LY_bundle_object is an FBT data object containing a bundle of LY geometry
    slices at different times, packaged within a single object (as opposed to
    a sequence of standalone LY files). LY_to_torax_times is a 1D array of
    times, defining the times in the TORAX simulation corresponding to each
    slice in the LY bundle. All times in the LY bundle must be mapped to
    times in TORAX. The LY_bundle_object and L_object can either be file names
    for disk loading, or directly the data dicts.

    Args:
      geometry_dir: Directory where to find the FBT file describing the magnetic
        geometry. If None, uses the environment variable TORAX_GEOMETRY_DIR if
        available. If that variable is not set and geometry_dir is not provided,
        then it defaults to another dir. See `load_geo_data` implementation.
      LY_bundle_object: Either file name for bundled LY data, e.g. as produced
        by liuqe meqlpack, or the data dict itself.
      L_object: Either file name for L data. Assumed to be the same L data for
        all LY slices in the bundle, or the data dict itself.
      LY_to_torax_times: User-provided times which map the times of the LY
        geometry slices to TORAX simulation times. A ValueError is raised if the
        number of array elements doesn't match the length of the LY_bundle array
        data. If None, then times are taken from the LY_bundle_object itself.
      Ip_from_parameters: If True, then Ip is taken from the config and the
        values in the Geometry are rescaled.
      n_rho: Grid resolution used for all TORAX cell variables.
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations.

    Returns:
      A mapping from user-provided (or inferred) times to
      StandardGeometryIntermediates instances based on the input slices. This
      can then be used to build a StandardGeometryProvider.
    """

    if isinstance(LY_bundle_object, str):
      LY_bundle = geometry_loader.load_geo_data(
          geometry_dir, LY_bundle_object, geometry_loader.GeometrySource.FBT
      )
    elif isinstance(LY_bundle_object, Mapping):
      LY_bundle = LY_bundle_object
    else:
      raise ValueError(
          'LY_bundle_object must be a string (file path) or a dictionary.'
      )

    if isinstance(L_object, str):
      L = geometry_loader.load_geo_data(
          geometry_dir, L_object, geometry_loader.GeometrySource.FBT
      )
    elif isinstance(L_object, Mapping):
      L = L_object
    else:
      raise ValueError('L_object must be a string (file path) or a dictionary.')

    # Raises a ValueError if the data is invalid.
    _validate_fbt_data(LY_bundle, L)

    if LY_to_torax_times is None:
      LY_to_torax_times = LY_bundle['t']  # ndarray of times
    else:
      if len(LY_to_torax_times) != len(LY_bundle['t']):
        raise ValueError(f"""
            Length of LY_to_torax_times must match length of LY bundle data:
            len(LY_to_torax_times)={len(LY_to_torax_times)},
            len(LY_bundle['t'])={len(LY_bundle['t'])}
            """)

    intermediates = {}
    for idx, t in enumerate(LY_to_torax_times):
      data_slice = cls._get_LY_single_slice_from_bundle(LY_bundle, idx)
      intermediates[t] = cls._from_fbt(
          data_slice, L, Ip_from_parameters, n_rho, hires_fac
      )

    return intermediates

  @classmethod
  def _get_LY_single_slice_from_bundle(
      cls,
      LY_bundle: Mapping[str, np.ndarray],
      idx: int,
  ) -> Mapping[str, np.ndarray]:
    """Returns a single LY slice from a bundled LY file, at index idx."""

    # The keys below are the relevant LY keys for the FBT geometry provider.
    relevant_keys = [
        'rBt',
        'aminor',
        'rgeom',
        'TQ',
        'FB',
        'FA',
        'Q1Q',
        'Q2Q',
        'Q3Q',
        'Q4Q',
        'Q5Q',
        'ItQ',
        'deltau',
        'deltal',
        'kappa',
        'FtPQ',
        'zA',
    ]
    LY_single_slice = {key: LY_bundle[key][..., idx] for key in relevant_keys}
    return LY_single_slice

  @classmethod
  def _from_fbt(
      cls,
      LY: Mapping[str, np.ndarray],
      L: Mapping[str, np.ndarray],
      Ip_from_parameters: bool = True,
      n_rho: int = 25,
      hires_fac: int = 4,
  ) -> StandardGeometryIntermediates:
    """Constructs a StandardGeometryIntermediates from a single FBT LY slice.

    Args:
      LY: A dictionary of relevant FBT LY geometry data.
      L: A dictionary of relevant FBT L geometry data.
      Ip_from_parameters: If True, then Ip is taken from the config and the
        values in the Geometry are rescaled.
      n_rho: Grid resolution used for all TORAX cell variables.
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations on initialization.

    Returns:
      A StandardGeometryIntermediates instance based on the input slice. This
      can then be used to build a StandardGeometry by passing to
      `build_standard_geometry`.
    """
    Rmaj = LY['rgeom'][-1]  # Major radius
    B0 = LY['rBt'] / Rmaj  # Vacuum toroidal magnetic field on axis
    Rmin = LY['aminor'][-1]  # Minor radius
    Phi = LY['FtPQ']  # Toroidal flux including plasma contribution
    rhon = np.sqrt(Phi / Phi[-1])  # Normalized toroidal flux coordinate
    psi = L['pQ'] ** 2 * (LY['FB'] - LY['FA']) + LY['FA']  # Poloidal flux
    # To avoid possible divisions by zero in diverted geometry. Value of what
    # replaces the zero does not matter, since it will be replaced by a spline
    # extrapolation in the post_init.
    LY_Q1Q = np.where(LY['Q1Q'] != 0, LY['Q1Q'], constants.CONSTANTS.eps)
    return cls(
        geometry_type=geometry.GeometryType.FBT,
        Ip_from_parameters=Ip_from_parameters,
        Rmaj=Rmaj,
        Rmin=Rmin,
        B=B0,
        psi=psi[0] - psi,
        Phi=Phi,
        Ip_profile=np.abs(LY['ItQ']),
        Rin=LY['rgeom'] - LY['aminor'],
        Rout=LY['rgeom'] + LY['aminor'],
        F=np.abs(LY['TQ']),
        int_dl_over_Bp=1 / LY_Q1Q,
        flux_surf_avg_1_over_R2=LY['Q2Q'],
        flux_surf_avg_Bp2=np.abs(LY['Q3Q']) / (4 * np.pi**2),
        flux_surf_avg_RBp=np.abs(LY['Q5Q']) / (2 * np.pi),
        flux_surf_avg_R2Bp2=np.abs(LY['Q4Q']) / (2 * np.pi) ** 2,
        delta_upper_face=LY['deltau'],
        delta_lower_face=LY['deltal'],
        elongation=LY['kappa'],
        vpr=4 * np.pi * Phi[-1] * rhon / (np.abs(LY['TQ']) * LY['Q2Q']),
        n_rho=n_rho,
        hires_fac=hires_fac,
        z_magnetic_axis=LY['zA'],
    )

  @classmethod
  def from_eqdsk(
      cls,
      geometry_dir: str | None,
      geometry_file: str,
      hires_fac: int,
      Ip_from_parameters: bool,
      n_rho: int,
      n_surfaces: int,
      last_surface_factor: float,
  ) -> StandardGeometryIntermediates:
    """Constructs a StandardGeometryIntermediates from EQDSK.

    This method constructs a StandardGeometryIntermediates object from an EQDSK
    file. It calculates flux surface averages based on the EQDSK geometry 2D psi
    mesh.

    Args:
      geometry_dir: Directory where to find the EQDSK file describing the
        magnetic geometry. If None, uses the environment variable
        TORAX_GEOMETRY_DIR if available. If that variable is not set and
        geometry_dir is not provided, then it defaults to another dir. See
        implementation.
      geometry_file: EQDSK file name.
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations.
      Ip_from_parameters: If True, then Ip is taken from the config and the
        values in the Geometry are rescaled.
      n_rho: Grid resolution used for all TORAX cell variables.
      n_surfaces: Number of surfaces for which flux surface averages are
        calculated.
      last_surface_factor: Multiplication factor of the boundary poloidal flux,
        used for the contour defining geometry terms at the LCFS on the TORAX
        grid. Needed to avoid divergent integrations in diverted geometries.

    Returns:
      A StandardGeometryIntermediates instance based on the input file. This
      can then be used to build a StandardGeometry by passing to
      `build_standard_geometry`.
    """

    def calculate_area(x, z):
      """Gauss-shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)."""
      n = len(x)
      area = 0.0
      for i in range(n):
        j = (i + 1) % n  # roll over at n
        area += x[i] * z[j]
        area -= z[i] * x[j]
      area = abs(area) / 2.0
      return area

    eqfile = geometry_loader.load_geo_data(
        geometry_dir, geometry_file, geometry_loader.GeometrySource.EQDSK
    )
    # TODO(b/375696414): deal with updown asymmetric cases.
    # Rmaj taken as Rgeo (LCFS Rmaj)
    Rmaj = (eqfile['xbdry'].max() + eqfile['xbdry'].min()) / 2.0
    Rmin = (eqfile['xbdry'].max() - eqfile['xbdry'].min()) / 2.0
    B0 = eqfile['bcentre']
    Raxis = eqfile['xmag']
    Zaxis = eqfile['zmag']

    # Set psi(axis) = 0
    psi_eqdsk_1dgrid = np.linspace(
        0.0, eqfile['psibdry'] - eqfile['psimag'], eqfile['nx']
    )

    X_1D = np.linspace(
        eqfile['xgrid1'], eqfile['xgrid1'] + eqfile['xdim'], eqfile['nx']
    )
    Z_1D = np.linspace(
        eqfile['zmid'] - eqfile['zdim'] / 2,
        eqfile['zmid'] + eqfile['zdim'] / 2,
        eqfile['nz'],
    )
    X, Z = np.meshgrid(X_1D, Z_1D, indexing='ij')
    Xlcfs, Zlcfs = eqfile['xbdry'], eqfile['zbdry']

    # Psi 2D grid defined on the Meshgrid. Set psi(axis) = 0
    psi_eqdsk_2dgrid = eqfile['psi'] - eqfile['psimag']
    # Create mask for the confined region, i.e.,Xlcfs.min() < X < Xlcfs.max(),
    # Zlcfs.min() < Z < Zlcfs.max()

    offset = 0.01
    mask = (
        (X > Xlcfs.min() - offset)
        & (X < Xlcfs.max() + offset)
        & (Z > Zlcfs.min() - offset)
        & (Z < Zlcfs.max() + offset)
    )
    masked_psi_eqdsk_2dgrid = np.ma.masked_where(~mask, psi_eqdsk_2dgrid)

    # q on uniform grid (pressure, etc., also defined here)
    q_eqdsk_1dgrid = eqfile['qpsi']

    # ---- Interpolations
    q_interp = scipy.interpolate.interp1d(
        psi_eqdsk_1dgrid, q_eqdsk_1dgrid, kind='cubic'
    )
    psi_spline_fit = scipy.interpolate.RectBivariateSpline(
        X_1D, Z_1D, psi_eqdsk_2dgrid, kx=3, ky=3, s=0
    )
    F_interp = scipy.interpolate.interp1d(
        psi_eqdsk_1dgrid, eqfile['fpol'], kind='cubic'
    )  # toroidal field flux function

    # -----------------------------------------------------------
    # --------- Make flux surface contours ---------
    # -----------------------------------------------------------

    psi_interpolant = np.linspace(
        0,
        (eqfile['psibdry'] - eqfile['psimag']) * last_surface_factor,
        n_surfaces,
    )

    surfaces = []
    cg_psi = contourpy.contour_generator(X, Z, masked_psi_eqdsk_2dgrid)

    # Skip magnetic axis since no contour is defined there.
    for _, _psi in enumerate(psi_interpolant[1:]):
      vertices = cg_psi.create_contour(_psi)
      if not vertices:
        raise ValueError(f"""
            Valid contour not found for EQDSK geometry for psi value {_psi}.
            Possible reason is too many surfaces requested.
            Try reducing n_surfaces from the current value of {n_surfaces}.
            """)
      x_surface, z_surface = vertices[0].T[0], vertices[0].T[1]
      surfaces.append((x_surface, z_surface))

    # -----------------------------------------------------------
    # --------- Compute Flux surface averages and 1D profiles ---------
    # --- Area, Volume, R_inboard, R_outboard
    # --- FSA: <1/R^2>, <Bp^2>, <|grad(psi)|>, <|grad(psi)|^2>
    # --- Toroidal plasma current
    # --- Integral dl/Bp
    # -----------------------------------------------------------

    # Gathering area for profiles
    areas, volumes = np.empty(len(surfaces) + 1), np.empty(len(surfaces) + 1)
    R_inboard, R_outboard = np.empty(len(surfaces) + 1), np.empty(
        len(surfaces) + 1
    )
    flux_surf_avg_1_over_R2_eqdsk = np.empty(len(surfaces) + 1)  # <1/R**2>
    flux_surf_avg_Bp2_eqdsk = np.empty(len(surfaces) + 1)  # <Bp**2>
    flux_surf_avg_RBp_eqdsk = np.empty(len(surfaces) + 1)  # <|grad(psi)|>
    flux_surf_avg_R2Bp2_eqdsk = np.empty(len(surfaces) + 1)  # <|grad(psi)|**2>
    int_dl_over_Bp_eqdsk = np.empty(
        len(surfaces) + 1
    )  # int(Rdl / | grad(psi) |)
    Ip_eqdsk = np.empty(len(surfaces) + 1)  # Toroidal plasma current
    delta_upper_face_eqdsk = np.empty(len(surfaces) + 1)  # Upper face delta
    delta_lower_face_eqdsk = np.empty(len(surfaces) + 1)  # Lower face delta
    elongation = np.empty(len(surfaces) + 1)  # Elongation

    # ---- Compute
    for n, (x_surface, z_surface) in enumerate(surfaces):

      # dl, line elements on which we will integrate
      surface_dl = np.sqrt(
          np.gradient(x_surface) ** 2 + np.gradient(z_surface) ** 2
      )

      # calculating gradient of psi in 2D
      surface_dpsi_x = psi_spline_fit.ev(x_surface, z_surface, dx=1)
      surface_dpsi_z = psi_spline_fit.ev(x_surface, z_surface, dy=1)
      surface_abs_grad_psi = np.sqrt(surface_dpsi_x**2 + surface_dpsi_z**2)

      # Poloidal field strength Bp = |grad(psi)| / R
      surface_Bpol = surface_abs_grad_psi / x_surface
      surface_int_dl_over_bpol = np.sum(
          surface_dl / surface_Bpol
      )  # This is denominator of all FSA

      # plasma current
      surface_int_bpol_dl = np.sum(surface_Bpol * surface_dl)

      # 4 FSA, < 1/ R^2>, < | grad psi | >, < B_pol^2>, < | grad psi |^2 >
      # where FSA(G) = int (G dl / Bpol) / (int (dl / Bpol))
      surface_FSA_int_one_over_r2 = (
          np.sum(1 / x_surface**2 * surface_dl / surface_Bpol)
          / surface_int_dl_over_bpol
      )
      surface_FSA_abs_grad_psi = (
          np.sum(surface_abs_grad_psi * surface_dl / surface_Bpol)
          / surface_int_dl_over_bpol
      )
      surface_FSA_Bpol_squared = (
          np.sum(surface_Bpol * surface_dl) / surface_int_dl_over_bpol
      )
      surface_FSA_abs_grad_psi2 = (
          np.sum(surface_abs_grad_psi**2 * surface_dl / surface_Bpol)
          / surface_int_dl_over_bpol
      )

      # volumes and areas
      area = calculate_area(x_surface, z_surface)
      volume = area * 2 * np.pi * Rmaj

      # Triangularity
      idx_upperextent = np.argmax(z_surface)
      idx_lowerextent = np.argmin(z_surface)

      Rmaj_local = (x_surface.max() + x_surface.min()) / 2.0
      Rmin_local = (x_surface.max() - x_surface.min()) / 2.0

      X_upperextent = x_surface[idx_upperextent]
      X_lowerextent = x_surface[idx_lowerextent]

      Z_upperextent = z_surface[idx_upperextent]
      Z_lowerextent = z_surface[idx_lowerextent]

      # (RMAJ - X_upperextent) / RMIN
      surface_delta_upper_face = (Rmaj_local - X_upperextent) / Rmin_local
      surface_delta_lower_face = (Rmaj_local - X_lowerextent) / Rmin_local

      # Append to lists.
      # Start with n=1 since n=0 is the magnetic axis with no contour defined.
      areas[n + 1] = area
      volumes[n + 1] = volume
      R_inboard[n + 1] = x_surface.min()
      R_outboard[n + 1] = x_surface.max()
      int_dl_over_Bp_eqdsk[n + 1] = surface_int_dl_over_bpol
      flux_surf_avg_1_over_R2_eqdsk[n + 1] = surface_FSA_int_one_over_r2
      flux_surf_avg_RBp_eqdsk[n + 1] = surface_FSA_abs_grad_psi
      flux_surf_avg_R2Bp2_eqdsk[n + 1] = surface_FSA_abs_grad_psi2
      flux_surf_avg_Bp2_eqdsk[n + 1] = surface_FSA_Bpol_squared
      Ip_eqdsk[n + 1] = surface_int_bpol_dl / constants.CONSTANTS.mu0
      delta_upper_face_eqdsk[n + 1] = surface_delta_upper_face
      delta_lower_face_eqdsk[n + 1] = surface_delta_lower_face
      elongation[n + 1] = (Z_upperextent - Z_lowerextent) / (2.0 * Rmin_local)

    # Now set n=0 quantities. StandardGeometryIntermediate values at the
    # magnetic axis are prescribed, since a contour cannot be defined there.
    areas[0] = 0
    volumes[0] = 0
    R_inboard[0] = Raxis
    R_outboard[0] = Raxis
    int_dl_over_Bp_eqdsk[0] = 0
    flux_surf_avg_1_over_R2_eqdsk[0] = 1 / Raxis**2
    flux_surf_avg_RBp_eqdsk[0] = 0
    flux_surf_avg_R2Bp2_eqdsk[0] = 0
    flux_surf_avg_Bp2_eqdsk[0] = 0
    Ip_eqdsk[0] = 0
    delta_upper_face_eqdsk[0] = delta_upper_face_eqdsk[1]
    delta_lower_face_eqdsk[0] = delta_lower_face_eqdsk[1]
    elongation[0] = elongation[1]

    # q-profile on interpolation
    q_profile = q_interp(psi_interpolant)

    # toroidal flux
    Phi_eqdsk = (
        scipy.integrate.cumulative_trapezoid(
            q_profile, psi_interpolant, initial=0.0
        )
        * 2
        * np.pi
    )

    # toroidal field flux function, T=RBphi
    F_eqdsk = F_interp(psi_interpolant)

    rhon = np.sqrt(Phi_eqdsk / Phi_eqdsk[-1])
    vpr = (
        4
        * np.pi
        * Phi_eqdsk[-1]
        * rhon
        / (F_eqdsk * flux_surf_avg_1_over_R2_eqdsk)
    )

    return cls(
        geometry_type=geometry.GeometryType.EQDSK,
        Ip_from_parameters=Ip_from_parameters,
        Rmaj=Rmaj,
        Rmin=Rmin,
        B=B0,
        # TODO(b/335204606): handle COCOS shenanigans
        psi=psi_interpolant * 2 * np.pi,
        Ip_profile=Ip_eqdsk,
        Phi=Phi_eqdsk,
        Rin=R_inboard,
        Rout=R_outboard,
        F=F_eqdsk,
        int_dl_over_Bp=int_dl_over_Bp_eqdsk,
        flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2_eqdsk,
        flux_surf_avg_RBp=flux_surf_avg_RBp_eqdsk,
        flux_surf_avg_R2Bp2=flux_surf_avg_R2Bp2_eqdsk,
        flux_surf_avg_Bp2=flux_surf_avg_Bp2_eqdsk,
        delta_upper_face=delta_upper_face_eqdsk,
        delta_lower_face=delta_lower_face_eqdsk,
        elongation=elongation,
        vpr=vpr,
        n_rho=n_rho,
        hires_fac=hires_fac,
        z_magnetic_axis=Zaxis,
    )


def build_standard_geometry(
    intermediate: StandardGeometryIntermediates,
) -> StandardGeometry:
  """Build geometry object based on set of profiles from an EQ solution.

  Args:
    intermediate: A StandardGeometryIntermediates object that holds the
      intermediate values used to build a StandardGeometry for this timeslice.
      These can either be direct or interpolated values.

  Returns:
    A StandardGeometry object.
  """

  # Toroidal flux coordinates
  rho_intermediate = np.sqrt(intermediate.Phi / (np.pi * intermediate.B))
  rho_norm_intermediate = rho_intermediate / rho_intermediate[-1]

  # flux surface integrals of various geometry quantities
  C1 = intermediate.int_dl_over_Bp

  C0 = intermediate.flux_surf_avg_RBp * C1
  C2 = intermediate.flux_surf_avg_1_over_R2 * C1
  C3 = intermediate.flux_surf_avg_Bp2 * C1
  C4 = intermediate.flux_surf_avg_R2Bp2 * C1

  # derived quantities for transport equations and transformations

  g0 = C0 * 2 * np.pi  # <\nabla psi> * (dV/dpsi), equal to <\nabla V>
  g1 = C1 * C4 * 4 * np.pi**2  # <(\nabla psi)**2> * (dV/dpsi) ** 2
  g2 = C1 * C3 * 4 * np.pi**2  # <(\nabla psi)**2 / R**2> * (dV/dpsi) ** 2
  g3 = C2[1:] / C1[1:]  # <1/R**2>
  g3 = np.concatenate((np.array([1 / intermediate.Rin[0] ** 2]), g3))
  g2g3_over_rhon = g2[1:] * g3[1:] / rho_norm_intermediate[1:]
  g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))

  # make an alternative initial psi, self-consistent with numerical geometry
  # Ip profile. Needed since input psi profile may have noisy second derivatives
  dpsidrhon = (
      intermediate.Ip_profile[1:]
      * (16 * constants.CONSTANTS.mu0 * np.pi**3 * intermediate.Phi[-1])
      / (g2g3_over_rhon[1:] * intermediate.F[1:])
  )
  dpsidrhon = np.concatenate((np.zeros(1), dpsidrhon))
  psi_from_Ip = scipy.integrate.cumulative_trapezoid(
      y=dpsidrhon, x=rho_norm_intermediate, initial=0.0
  )

  # set Ip-consistent psi derivative boundary condition (although will be
  # replaced later with an fvm constraint)
  psi_from_Ip[-1] = psi_from_Ip[-2] + (
      16 * constants.CONSTANTS.mu0 * np.pi**3 * intermediate.Phi[-1]
  ) * intermediate.Ip_profile[-1] / (
      g2g3_over_rhon[-1] * intermediate.F[-1]
  ) * (
      rho_norm_intermediate[-1] - rho_norm_intermediate[-2]
  )

  # dV/drhon, dS/drhon
  vpr = intermediate.vpr
  spr = vpr / (2 * np.pi * intermediate.Rmaj)

  # Volume and area
  volume_intermediate = scipy.integrate.cumulative_trapezoid(
      y=vpr, x=rho_norm_intermediate, initial=0.0
  )
  area_intermediate = volume_intermediate / (2 * np.pi * intermediate.Rmaj)

  # plasma current density
  dI_tot_drhon = np.gradient(intermediate.Ip_profile, rho_norm_intermediate)

  jtot_face_bulk = dI_tot_drhon[1:] / spr[1:]

  # For now set on-axis to the same as the second grid point, due to 0/0
  # division.
  jtot_face_axis = jtot_face_bulk[0]

  jtot = np.concatenate([np.array([jtot_face_axis]), jtot_face_bulk])

  # fill geometry structure
  drho_norm = float(rho_norm_intermediate[-1]) / intermediate.n_rho
  # normalized grid
  mesh = torax_pydantic.Grid1D.construct(nx=intermediate.n_rho, dx=drho_norm)
  rho_b = rho_intermediate[-1]  # radius denormalization constant
  # helper variables for mesh cells and faces
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current.
  rho_hires_norm = np.linspace(
      0, 1, intermediate.n_rho * intermediate.hires_fac
  )
  rho_hires = rho_hires_norm * rho_b

  rhon_interpolation_func = lambda x, y: np.interp(x, rho_norm_intermediate, y)
  # V' for volume integrations on face grid
  vpr_face = rhon_interpolation_func(rho_face_norm, vpr)
  # V' for volume integrations on cell grid
  vpr = rhon_interpolation_func(rho_norm, vpr)

  # S' for area integrals on face grid
  spr_face = rhon_interpolation_func(rho_face_norm, spr)
  # S' for area integrals on cell grid
  spr_cell = rhon_interpolation_func(rho_norm, spr)
  spr_hires = rhon_interpolation_func(rho_hires_norm, spr)

  # triangularity on cell grid
  delta_upper_face = rhon_interpolation_func(
      rho_face_norm, intermediate.delta_upper_face
  )
  delta_lower_face = rhon_interpolation_func(
      rho_face_norm, intermediate.delta_lower_face
  )

  # average triangularity
  delta_face = 0.5 * (delta_upper_face + delta_lower_face)

  # elongation
  elongation = rhon_interpolation_func(rho_norm, intermediate.elongation)
  elongation_face = rhon_interpolation_func(
      rho_face_norm, intermediate.elongation
  )

  Phi_face = rhon_interpolation_func(rho_face_norm, intermediate.Phi)
  Phi = rhon_interpolation_func(rho_norm, intermediate.Phi)

  F_face = rhon_interpolation_func(rho_face_norm, intermediate.F)
  F = rhon_interpolation_func(rho_norm, intermediate.F)
  F_hires = rhon_interpolation_func(rho_hires_norm, intermediate.F)

  psi = rhon_interpolation_func(rho_norm, intermediate.psi)
  psi_from_Ip_face = rhon_interpolation_func(rho_face_norm, psi_from_Ip)
  psi_from_Ip = rhon_interpolation_func(rho_norm, psi_from_Ip)

  jtot_face = rhon_interpolation_func(rho_face_norm, jtot)
  jtot = rhon_interpolation_func(rho_norm, jtot)

  Ip_profile_face = rhon_interpolation_func(
      rho_face_norm, intermediate.Ip_profile
  )

  Rin_face = rhon_interpolation_func(rho_face_norm, intermediate.Rin)
  Rin = rhon_interpolation_func(rho_norm, intermediate.Rin)

  Rout_face = rhon_interpolation_func(rho_face_norm, intermediate.Rout)
  Rout = rhon_interpolation_func(rho_norm, intermediate.Rout)

  g0_face = rhon_interpolation_func(rho_face_norm, g0)
  g0 = rhon_interpolation_func(rho_norm, g0)

  g1_face = rhon_interpolation_func(rho_face_norm, g1)
  g1 = rhon_interpolation_func(rho_norm, g1)

  g2_face = rhon_interpolation_func(rho_face_norm, g2)
  g2 = rhon_interpolation_func(rho_norm, g2)

  g3_face = rhon_interpolation_func(rho_face_norm, g3)
  g3 = rhon_interpolation_func(rho_norm, g3)

  g2g3_over_rhon_face = rhon_interpolation_func(rho_face_norm, g2g3_over_rhon)
  g2g3_over_rhon_hires = rhon_interpolation_func(rho_hires_norm, g2g3_over_rhon)
  g2g3_over_rhon = rhon_interpolation_func(rho_norm, g2g3_over_rhon)

  volume_face = rhon_interpolation_func(rho_face_norm, volume_intermediate)
  volume = rhon_interpolation_func(rho_norm, volume_intermediate)

  area_face = rhon_interpolation_func(rho_face_norm, area_intermediate)
  area = rhon_interpolation_func(rho_norm, area_intermediate)

  return StandardGeometry(
      geometry_type=intermediate.geometry_type,
      torax_mesh=mesh,
      Phi=Phi,
      Phi_face=Phi_face,
      Rmaj=intermediate.Rmaj,
      Rmin=intermediate.Rmin,
      B0=intermediate.B,
      volume=volume,
      volume_face=volume_face,
      area=area,
      area_face=area_face,
      vpr=vpr,
      vpr_face=vpr_face,
      spr=spr_cell,
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
      Ip_from_parameters=intermediate.Ip_from_parameters,
      Ip_profile_face=Ip_profile_face,
      psi=psi,
      psi_from_Ip=psi_from_Ip,
      psi_from_Ip_face=psi_from_Ip_face,
      jtot=jtot,
      jtot_face=jtot_face,
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
      elongation=elongation,
      elongation_face=elongation_face,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
      _z_magnetic_axis=intermediate.z_magnetic_axis,
  )


def _validate_fbt_data(
    LY: Mapping[str, np.ndarray], L: Mapping[str, np.ndarray]
) -> None:
  """Validates the FBT data dictionaries.

  Works for both single slice and bundle LY data.

  Args:
    LY: A dictionary of FBT LY geometry data.
    L: A dictionary of FBT L geometry data.

  Raises a ValueError if the data is invalid.
  """

  # The checks for L['pQ'] and LY['t'] are done first since their existence
  # is needed for the shape checks.
  if 'pQ' not in L:
    raise ValueError("L data is missing the 'pQ' key.")
  if 't' not in LY:
    raise ValueError("LY data is missing the 't' key.")

  len_psinorm = len(L['pQ'])
  len_times = len(LY['t']) if LY['t'].shape else 1  # Handle scalar t
  time_only_shape = (len_times,) if len_times > 1 else ()
  psi_and_time_shape = (
      (len_psinorm, len_times) if len_times > 1 else (len_psinorm,)
  )

  required_LY_spec = {
      'rBt': time_only_shape,
      'aminor': psi_and_time_shape,
      'rgeom': psi_and_time_shape,
      'TQ': psi_and_time_shape,
      'FB': time_only_shape,
      'FA': time_only_shape,
      'Q1Q': psi_and_time_shape,
      'Q2Q': psi_and_time_shape,
      'Q3Q': psi_and_time_shape,
      'Q4Q': psi_and_time_shape,
      'Q5Q': psi_and_time_shape,
      'ItQ': psi_and_time_shape,
      'deltau': psi_and_time_shape,
      'deltal': psi_and_time_shape,
      'kappa': psi_and_time_shape,
      'FtPQ': psi_and_time_shape,
      'zA': time_only_shape,
  }

  missing_LY_keys = required_LY_spec.keys() - LY.keys()
  if missing_LY_keys:
    raise ValueError(
        f'LY data is missing the following keys: {missing_LY_keys}'
    )

  for key, shape in required_LY_spec.items():
    if LY[key].shape != shape:
      raise ValueError(
          f"Incorrect shape for key '{key}' in LY data. "
          f'Expected {shape}:, got {LY[key].shape}.'
      )


# TODO(b/401502047): Investigate how window_length should depend on the
# resolution of the data.
def _smooth_savgol(
    data: np.ndarray,
    idx_limit: int,
    polyorder: int,
    window_length: int = 5,
    preserve_first: bool = True,
) -> np.ndarray:
  """Smooths data using Savitzky-Golay polynomial filter.

  Args:
    data: 1D array of data to be smoothed.
    idx_limit: Index up to which the smoothing is applied.
    polyorder: Polynomial order of the Savitzky-Golay filter.
    window_length: Window length of the Savitzky-Golay filter.
    preserve_first: If True, the first data point is preserved, otherwise it is
      smoothed.

  Returns:
    Smoothed data array. No-op if idx_limit is 0 (no smoothing).
  """
  if idx_limit == 0:
    return data
  smoothed_data = scipy.signal.savgol_filter(
      data, window_length, polyorder, mode='nearest'
  )
  first_point = data[0] if preserve_first else smoothed_data[0]
  return np.concatenate(
      [np.array([first_point]), smoothed_data[1:idx_limit], data[idx_limit:]]
  )
