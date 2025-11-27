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
import dataclasses

import chex
import jax
from jax import numpy as jnp
import numpy as np
import scipy
from torax._src import array_typing
from torax._src import constants
from torax._src import interpolated_param
from torax._src import jax_utils
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

_RHO_SMOOTHING_LIMIT = 0.1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometry(geometry.Geometry):
  r"""Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.  This class extends
  the base `Geometry` class with attributes that are commonly computed
  from various equilibrium data sources (CHEASE, FBT, EQDSK, etc.).

  Attributes:
    Ip_from_parameters: Boolean indicating whether the total plasma current
      (`Ip`) is determined by the config parameters (True) or read from the
      geometry file (False). This field is marked as static and will retrigger
      compilation if changed.
    Ip_profile_face: Plasma current profile on the face grid
      [:math:`\mathrm{A}`].
    psi: 1D poloidal flux profile on the cell grid [:math:`\mathrm{Wb}`].
    psi_from_Ip: Poloidal flux profile on the cell grid  [:math:`\mathrm{Wb}`],
      calculated from the plasma current profile in the geometry file.
    psi_from_Ip_face: Poloidal flux profile on the face grid [Wb], calculated
      from the plasma current profile in the geometry file.
    j_total: Total toroidal current density profile on the cell grid
      [:math:`\mathrm{A/m^2}`].
    j_total_face: Total toroidal current density profile on the face grid
      [:math:`\mathrm{A/m^2}`].
    delta_upper_face: Upper triangularity on the face grid [dimensionless]. See
      `Geometry` docstring for definition of `delta_upper_face`.
    delta_lower_face: Lower triangularity on the face grid [dimensionless]. See
      `Geometry` docstring for definition of `delta_lower_face`.
    connection_length_target: Optional input. Parallel connection length from
      outboard midplane to target [m]. If not provided, then the value is taken
      from runtime parameters in any model that needs it, e.g. edge models.
    connection_length_divertor: Optional input. Parallel connection length from
      outboard midplane to X-point [m]. If not provided, same procedure holds as
      for `connection_length_target`.
    target_angle_of_incidence: Optional input. Angle between magnetic field line
      and divertor target [degrees]. If not provided, same procedure holds as
      for `connection_length_target`.
    R_OMP: Optional input. Major radius of the outboard midplane [m]. If not
      provided, same procedure holds as for `connection_length_target`.
    R_target: Optional input. Major radius of the divertor target strike point
      [m]. If not provided, same procedure holds as for
      `connection_length_target`.
    B_pol_OMP: Optional input. Poloidal magnetic field at the outboard midplane
      [T]. If not provided, same procedure holds as for
      `connection_length_target`.
    diverted: Optional input. Boolean flag indicating whether the geometry is
      diverted. If not available, then diverted/limited will be determined by
      flux_surf_avg_Bp2[-1] in StandardGeometryIntermediates.
  """

  Ip_from_parameters: bool = dataclasses.field(metadata=dict(static=True))
  Ip_profile_face: array_typing.FloatVectorFace
  psi: array_typing.FloatVectorCell
  psi_from_Ip: array_typing.FloatVectorCell
  psi_from_Ip_face: array_typing.FloatVectorFace
  j_total: array_typing.Array
  j_total_face: array_typing.FloatVectorFace
  delta_upper_face: array_typing.FloatVectorFace
  delta_lower_face: array_typing.FloatVectorFace
  # Optional parameters not present in all geometry sources.
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  target_angle_of_incidence: array_typing.FloatScalar | None
  R_OMP: array_typing.FloatScalar | None
  R_target: array_typing.FloatScalar | None
  B_pol_OMP: array_typing.FloatScalar | None
  diverted: array_typing.BoolScalar | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometryProvider(geometry_provider.TimeDependentGeometryProvider):
  """Values to be interpolated for a Standard Geometry."""

  Ip_from_parameters: bool = dataclasses.field(metadata=dict(static=True))
  Ip_profile_face: interpolated_param.InterpolatedVarSingleAxis
  psi: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip_face: interpolated_param.InterpolatedVarSingleAxis
  j_total: interpolated_param.InterpolatedVarSingleAxis
  j_total_face: interpolated_param.InterpolatedVarSingleAxis
  delta_upper_face: interpolated_param.InterpolatedVarSingleAxis
  delta_lower_face: interpolated_param.InterpolatedVarSingleAxis
  elongation: interpolated_param.InterpolatedVarSingleAxis
  elongation_face: interpolated_param.InterpolatedVarSingleAxis
  connection_length_target: interpolated_param.InterpolatedVarSingleAxis | None
  connection_length_divertor: (
      interpolated_param.InterpolatedVarSingleAxis | None
  )
  target_angle_of_incidence: interpolated_param.InterpolatedVarSingleAxis | None
  R_OMP: interpolated_param.InterpolatedVarSingleAxis | None
  R_target: interpolated_param.InterpolatedVarSingleAxis | None
  B_pol_OMP: interpolated_param.InterpolatedVarSingleAxis | None
  diverted: interpolated_param.InterpolatedVarSingleAxis | None

  def __call__(self, t: chex.Numeric) -> geometry.Geometry:
    """Returns a Geometry instance at the given time."""
    chex.assert_type(t, jnp.floating)
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
    R_major: major radius on the magnetic axis in [:math:`\mathrm{m}`].
    a_minor: minor radius (a) in [:math:`\mathrm{m}`].
    B_0: Toroidal magnetic field on axis [:math:`\mathrm{T}`].
    psi: Poloidal flux profile [:math:`\mathrm{Wb}`].
    Ip_profile: Plasma current profile [:math:`\mathrm{A}`].
    Phi: Toroidal flux profile [:math:`\mathrm{Wb}`].
    R_in: Radius of the flux surface at the inboard side at midplane
      [:math:`\mathrm{m}`]. Inboard side is defined as the innermost radius.
    R_out: Radius of the flux surface at the outboard side at midplane
      [:math:`\mathrm{m}`]. Outboard side is defined as the outermost radius.
    F: Toroidal field flux function (:math:`F = R B_{\phi}`) [:math:`\mathrm{m
      T}`].
    int_dl_over_Bp: :math:`\oint dl/B_p` (field-line contour integral on the
      flux surface) [:math:`\mathrm{m / T}`], where :math:`B_p` is the poloidal
      magnetic field.
    flux_surf_avg_1_over_R: Flux surface average of :math:`1/R`
      [:math:`\mathrm{m^{-1}}`].
    flux_surf_avg_1_over_R2: Flux surface average of :math:`1/R^2`
      [:math:`\mathrm{m^{-2}}`].
    flux_surf_avg_grad_psi: Flux surface average of :math:`|\nabla \psi| = R
      B_p` [:math:`\mathrm{m T}`] (COCOS11).
    flux_surf_avg_grad_psi2: Flux surface average of :math:`|\nabla \psi|^2 =
      R^2 B_p^2` [:math:`\mathrm{m^2 T^2}`] (COCOS11).
    flux_surf_avg_grad_psi2_over_R2: Flux surface average of :math:`|\nabla
      \psi|^2 / R^2 = B_p^2` [:math:`\mathrm{T^2}`] (COCOS11).
    flux_surf_avg_B2: Flux surface average of :math:`B^2`
      [:math:`\mathrm{T}^2`].
    flux_surf_avg_1_over_B2: Flux surface average of :math:`1/B^2`
      [:math:`\mathrm{T}^{-2}`].
    delta_upper_face: Upper triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    delta_lower_face: Lower triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    elongation: Plasma elongation profile [dimensionless]. See `Geometry`
      docstring for definition.
    vpr:  Profile of dVolume/d(rho_norm), where rho_norm is the normalized
      toroidal flux coordinate [:math:`\mathrm{m^3}`].
    n_rho: Radial grid points (number of cells).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations. Used to create a higher-resolution grid to improve accuracy
      when initializing psi from a plasma current profile.
    z_magnetic_axis: z position of magnetic axis [:math:`\mathrm{m}`].
    diverted: Boolean flag indicating whether the geometry is diverted.
    connection_length_target: Parallel connection length from outboard midplane
      to target [m].
    connection_length_divertor: Parallel connection length from outboard
      midplane to X-point [m].
    target_angle_of_incidence: Angle between magnetic field line and divertor
      target [degrees].
    R_OMP: Major radius of the outboard midplane [m].
    R_target: Major radius of the divertor target strike point [m].
    B_pol_OMP: Poloidal magnetic field at the outboard midplane [T].
  """

  geometry_type: geometry.GeometryType
  Ip_from_parameters: bool
  R_major: array_typing.FloatScalar
  a_minor: array_typing.FloatScalar
  B_0: array_typing.FloatScalar
  psi: array_typing.Array
  Ip_profile: array_typing.Array
  Phi: array_typing.Array
  R_in: array_typing.Array
  R_out: array_typing.Array
  F: array_typing.Array
  int_dl_over_Bp: array_typing.Array
  flux_surf_avg_1_over_R: array_typing.Array
  flux_surf_avg_1_over_R2: array_typing.Array
  flux_surf_avg_grad_psi: array_typing.Array
  flux_surf_avg_grad_psi2: array_typing.Array
  flux_surf_avg_grad_psi2_over_R2: array_typing.Array
  flux_surf_avg_B2: array_typing.Array
  flux_surf_avg_1_over_B2: array_typing.Array
  delta_upper_face: array_typing.Array
  delta_lower_face: array_typing.Array
  elongation: array_typing.Array
  vpr: array_typing.Array
  n_rho: int
  hires_factor: int
  z_magnetic_axis: array_typing.FloatScalar | None
  # Optional parameters not present in all geometry sources.
  diverted: bool | None
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  target_angle_of_incidence: array_typing.FloatScalar | None
  R_OMP: array_typing.FloatScalar | None
  R_target: array_typing.FloatScalar | None
  B_pol_OMP: array_typing.FloatScalar | None

  def __post_init__(self):
    """Extrapolates edge values and smooths near-axis values.

    - Edge extrapolation for a subset of attributes based on a Cubic spline fit.
    - Near-axis smoothing for a subset of attributes based on a Savitzky-Golay
      filter with an appropriate polynominal order based on the attribute.
    """

    # Check if last flux surface is diverted and correct via spline fit if so
    if self.diverted or self.flux_surf_avg_grad_psi2_over_R2[-1] < 1e-10:
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
      flux_surf_avg_grad_psi2_over_R2_edge = spline(
          rhon,
          self.flux_surf_avg_grad_psi2_over_R2,
          1.0,
          bc_type='not-a-knot',
      )
      int_dl_over_Bp_edge = spline(
          rhon,
          self.int_dl_over_Bp,
          1.0,
          bc_type='not-a-knot',
      )
      g2_edge_ratio = (
          flux_surf_avg_grad_psi2_over_R2_edge * int_dl_over_Bp_edge**2
      ) / (
          self.flux_surf_avg_grad_psi2_over_R2[-2]
          * self.int_dl_over_Bp[-2] ** 2
      )
      if g2_edge_ratio > 1.0:
        bc_type = 'not-a-knot'
      else:
        bc_type = 'natural'
      set_edge = lambda array: spline(rhon, array, 1.0, bc_type)
      self.int_dl_over_Bp[-1] = set_edge(self.int_dl_over_Bp)
      self.flux_surf_avg_grad_psi2_over_R2[-1] = set_edge(
          self.flux_surf_avg_grad_psi2_over_R2
      )
      self.flux_surf_avg_1_over_R2[-1] = set_edge(self.flux_surf_avg_1_over_R2)
      self.flux_surf_avg_grad_psi[-1] = set_edge(self.flux_surf_avg_grad_psi)
      self.flux_surf_avg_grad_psi2[-1] = set_edge(self.flux_surf_avg_grad_psi2)
      self.vpr[-1] = set_edge(self.vpr)

    # Near-axis smoothing of quantities with known near-axis trends with rho
    rhon = np.sqrt(self.Phi / self.Phi[-1])
    idx_limit = np.argmin(np.abs(rhon - _RHO_SMOOTHING_LIMIT))

    # Bp goes like rho near-axis. So Bp2 terms are smoothed with order 2,
    # and Bp terms with order 1. vpr also goes like rho near-axis
    self.flux_surf_avg_grad_psi2_over_R2[:] = _smooth_savgol(
        self.flux_surf_avg_grad_psi2_over_R2, idx_limit, 2
    )
    self.flux_surf_avg_grad_psi2[:] = _smooth_savgol(
        self.flux_surf_avg_grad_psi2, idx_limit, 2
    )
    self.flux_surf_avg_grad_psi[:] = _smooth_savgol(
        self.flux_surf_avg_grad_psi, idx_limit, 1
    )
    self.vpr[:] = _smooth_savgol(self.vpr, idx_limit, 1)


def build_standard_geometry(
    intermediates: StandardGeometryIntermediates,
) -> StandardGeometry:
  """Build geometry object based on set of profiles from an EQ solution.

  Args:
    intermediates: A StandardGeometryIntermediates object that holds the
      intermediate values used to build a StandardGeometry for this timeslice.
      These can either be direct or interpolated values.

  Returns:
    A StandardGeometry object.
  """

  # Toroidal flux coordinates
  rho_intermediate = np.sqrt(intermediates.Phi / (np.pi * intermediates.B_0))
  rho_norm_intermediate = rho_intermediate / rho_intermediate[-1]

  # derived geometric quantities
  dV_dpsi = intermediates.int_dl_over_Bp
  g0 = intermediates.flux_surf_avg_grad_psi * dV_dpsi  # <\nabla V>
  g1 = intermediates.flux_surf_avg_grad_psi2 * dV_dpsi**2  # <(\nabla V)^2>
  g2 = (
      intermediates.flux_surf_avg_grad_psi2_over_R2 * dV_dpsi**2
  )  # <(\nabla V)^2 / R^2>
  g3 = intermediates.flux_surf_avg_1_over_R2  # <1/R**2>
  g2g3_over_rhon = g2[1:] * g3[1:] / rho_norm_intermediate[1:]
  g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))

  # make an alternative initial psi, self-consistent with numerical geometry
  # Ip profile. Needed since input psi profile may have noisy second derivatives
  dpsidrhon = (
      intermediates.Ip_profile[1:]
      * (16 * constants.CONSTANTS.mu_0 * np.pi**3 * intermediates.Phi[-1])
      / (g2g3_over_rhon[1:] * intermediates.F[1:])
  )
  dpsidrhon = np.concatenate((np.zeros(1), dpsidrhon))
  psi_from_Ip = scipy.integrate.cumulative_trapezoid(
      y=dpsidrhon,
      x=rho_norm_intermediate,
      initial=0.0,
  )
  # `initial` can only be zero or None, so add psi_axis afterwards.
  psi_from_Ip += intermediates.psi[0]

  # set Ip-consistent psi derivative boundary condition (although will be
  # replaced later with an fvm constraint)
  psi_from_Ip[-1] = psi_from_Ip[-2] + (
      16 * constants.CONSTANTS.mu_0 * np.pi**3 * intermediates.Phi[-1]
  ) * intermediates.Ip_profile[-1] / (
      g2g3_over_rhon[-1] * intermediates.F[-1]
  ) * (
      rho_norm_intermediate[-1] - rho_norm_intermediate[-2]
  )

  # dV/drhon, dS/drhon
  vpr = intermediates.vpr
  spr = vpr * intermediates.flux_surf_avg_1_over_R / (2 * np.pi)

  # Volume and area
  volume_intermediate = scipy.integrate.cumulative_trapezoid(
      y=vpr, x=rho_norm_intermediate, initial=0.0
  )
  area_intermediate = scipy.integrate.cumulative_trapezoid(
      y=spr, x=rho_norm_intermediate, initial=0.0
  )

  # plasma current density
  dI_tot_drhon = np.gradient(intermediates.Ip_profile, rho_norm_intermediate)

  j_total_face_bulk = dI_tot_drhon[1:] / spr[1:]

  # For now set on-axis to the same as the second grid point, due to 0/0
  # division.
  j_total_face_axis = j_total_face_bulk[0]

  j_total = np.concatenate([np.array([j_total_face_axis]), j_total_face_bulk])

  # fill geometry structure
  # normalized grid
  mesh = torax_pydantic.Grid1D(nx=intermediates.n_rho)
  rho_b = rho_intermediate[-1]  # radius denormalization constant
  # helper variables for mesh cells and faces
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current.
  rho_hires_norm = np.linspace(
      0, 1, intermediates.n_rho * intermediates.hires_factor
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
      rho_face_norm, intermediates.delta_upper_face
  )
  delta_lower_face = rhon_interpolation_func(
      rho_face_norm, intermediates.delta_lower_face
  )

  # average triangularity
  delta_face = 0.5 * (delta_upper_face + delta_lower_face)

  # elongation
  elongation = rhon_interpolation_func(rho_norm, intermediates.elongation)
  elongation_face = rhon_interpolation_func(
      rho_face_norm, intermediates.elongation
  )

  Phi_face = rhon_interpolation_func(rho_face_norm, intermediates.Phi)
  Phi = rhon_interpolation_func(rho_norm, intermediates.Phi)

  F_face = rhon_interpolation_func(rho_face_norm, intermediates.F)
  F = rhon_interpolation_func(rho_norm, intermediates.F)
  F_hires = rhon_interpolation_func(rho_hires_norm, intermediates.F)

  psi = rhon_interpolation_func(rho_norm, intermediates.psi)
  psi_from_Ip_face = rhon_interpolation_func(rho_face_norm, psi_from_Ip)
  psi_from_Ip = rhon_interpolation_func(rho_norm, psi_from_Ip)

  j_total_face = rhon_interpolation_func(rho_face_norm, j_total)
  j_total = rhon_interpolation_func(rho_norm, j_total)

  Ip_profile_face = rhon_interpolation_func(
      rho_face_norm, intermediates.Ip_profile
  )

  Rin_face = rhon_interpolation_func(rho_face_norm, intermediates.R_in)
  Rin = rhon_interpolation_func(rho_norm, intermediates.R_in)

  Rout_face = rhon_interpolation_func(rho_face_norm, intermediates.R_out)
  Rout = rhon_interpolation_func(rho_norm, intermediates.R_out)

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

  gm4 = rhon_interpolation_func(rho_norm, intermediates.flux_surf_avg_1_over_B2)
  gm4_face = rhon_interpolation_func(
      rho_face_norm, intermediates.flux_surf_avg_1_over_B2
  )
  gm5 = rhon_interpolation_func(rho_norm, intermediates.flux_surf_avg_B2)
  gm5_face = rhon_interpolation_func(
      rho_face_norm, intermediates.flux_surf_avg_B2
  )

  volume_face = rhon_interpolation_func(rho_face_norm, volume_intermediate)
  volume = rhon_interpolation_func(rho_norm, volume_intermediate)

  area_face = rhon_interpolation_func(rho_face_norm, area_intermediate)
  area = rhon_interpolation_func(rho_norm, area_intermediate)

  return StandardGeometry(
      geometry_type=intermediates.geometry_type,
      torax_mesh=mesh,
      Phi=Phi,
      Phi_face=Phi_face,
      R_major=intermediates.R_major,
      a_minor=intermediates.a_minor,
      B_0=intermediates.B_0,
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
      gm4=gm4,
      gm4_face=gm4_face,
      gm5=gm5,
      gm5_face=gm5_face,
      F=F,
      F_face=F_face,
      F_hires=F_hires,
      R_in=Rin,
      R_in_face=Rin_face,
      R_out=Rout,
      R_out_face=Rout_face,
      Ip_from_parameters=intermediates.Ip_from_parameters,
      Ip_profile_face=Ip_profile_face,
      psi=psi,
      psi_from_Ip=psi_from_Ip,
      psi_from_Ip_face=psi_from_Ip_face,
      j_total=j_total,
      j_total_face=j_total_face,
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
      Phi_b_dot=np.zeros((), dtype=jax_utils.get_int_dtype()),
      _z_magnetic_axis=intermediates.z_magnetic_axis,
      diverted=intermediates.diverted,
      connection_length_target=intermediates.connection_length_target,
      connection_length_divertor=intermediates.connection_length_divertor,
      target_angle_of_incidence=intermediates.target_angle_of_incidence,
      R_OMP=intermediates.R_OMP,
      R_target=intermediates.R_target,
      B_pol_OMP=intermediates.B_pol_OMP,
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
