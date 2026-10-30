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
CHEASE, MEQ, etc.
"""

import dataclasses
import functools

from absl import logging
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
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

_RHO_SMOOTHING_LIMIT = 0.1
_COARSE_GRID_THRESHOLD = 0.02
_COARSE_GRID_WARNING_INTERVAL_SEC = 10 * 60
# Radial profiles that vanish on the magnetic axis.
_ZERO_ON_AXIS = frozenset({
    'I_p_profile',
    'Phi',
    'flux_surf_avg_grad_psi',
    'flux_surf_avg_grad_psi2',
    'flux_surf_avg_grad_psi2_over_R2',
    'delta_upper_face',
    'delta_lower_face',
    'vpr',
    'trapped_fraction',
})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometry(geometry.Geometry):
  r"""Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.  This class extends
  the base `Geometry` class with attributes that are commonly computed
  from various equilibrium data sources (CHEASE, MEQ, EQDSK, etc.).

  Attributes:
    I_p_profile_face: Plasma current profile on the face grid
      [:math:`\mathrm{A}`].
    psi: 1D poloidal flux profile on the cell grid [:math:`\mathrm{Wb}`].
    psi_from_I_p: Poloidal flux profile on the cell grid [:math:`\mathrm{Wb}`],
      calculated from the plasma current profile in the geometry file.
    psi_from_I_p_face: Poloidal flux profile on the face grid [Wb], calculated
      from the plasma current profile in the geometry file.
    j_toroidal_total: Total toroidal current density profile on the cell grid
      [:math:`\mathrm{A/m^2}`].
    j_toroidal_total_face: Total toroidal current density profile on the face
      grid [:math:`\mathrm{A/m^2}`].
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
    angle_of_incidence_target: Optional input. Angle between magnetic field line
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
    r0_vacuum: Optional input. Reference major radius at which the input
      vacuum toroidal field was given [m], e.g. IMAS
      ``vacuum_toroidal_field.r0``. Used to write the vacuum field back at the
      same reference radius. If not provided, ``R_major`` is used.
  """

  I_p_profile_face: array_typing.FloatVectorFace
  psi: array_typing.FloatVectorCell
  psi_from_I_p: array_typing.FloatVectorCell
  psi_from_I_p_face: array_typing.FloatVectorFace
  j_toroidal_total: array_typing.Array
  j_toroidal_total_face: array_typing.FloatVectorFace
  delta_upper_face: array_typing.FloatVectorFace
  delta_lower_face: array_typing.FloatVectorFace
  # Optional parameters not present in all geometry sources.
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  angle_of_incidence_target: array_typing.FloatScalar | None
  R_OMP: array_typing.FloatScalar | None
  R_target: array_typing.FloatScalar | None
  B_pol_OMP: array_typing.FloatScalar | None
  diverted: array_typing.BoolScalar | None
  r0_vacuum: array_typing.FloatScalar | None = None

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Serializes stacked StandardGeometry into a flat dictionary of OutputVar tuples."""
    output_dict = super().to_output_dict(context)
    vars_to_pack: dict[output_keys.OutputKey, chex.Numeric] = {
        output_keys.I_P_PROFILE_FROM_GEO: self.I_p_profile_face,
        output_keys.PSI_FROM_GEO: self.psi,
        output_keys.PSI_FROM_I_P: (
            output_grid_context.extend_cell_grid_to_boundaries(
                self.psi_from_I_p, self.psi_from_I_p_face
            )
        ),
        output_keys.DELTA_UPPER: self.delta_upper_face,
        output_keys.DELTA_LOWER: self.delta_lower_face,
    }
    optional_scalars = (
        (output_keys.CONNECTION_LENGTH_TARGET, self.connection_length_target),
        (
            output_keys.CONNECTION_LENGTH_DIVERTOR,
            self.connection_length_divertor,
        ),
        (output_keys.ANGLE_OF_INCIDENCE_TARGET, self.angle_of_incidence_target),
        (output_keys.R_OMP, self.R_OMP),
        (output_keys.R_TARGET, self.R_target),
        (output_keys.B_POL_OMP, self.B_pol_OMP),
        (output_keys.DIVERTED, self.diverted),
    )
    for key, value in optional_scalars:
      if value is not None:
        vars_to_pack[key] = value

    for key, val in vars_to_pack.items():
      output_dict[key] = context.pack(key, val)
    return output_dict


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometryProvider(geometry_provider.TimeDependentGeometryProvider):
  """Values to be interpolated for a Standard Geometry."""

  I_p_profile_face: interpolated_param.InterpolatedVarSingleAxis
  psi: interpolated_param.InterpolatedVarSingleAxis
  psi_from_I_p: interpolated_param.InterpolatedVarSingleAxis
  psi_from_I_p_face: interpolated_param.InterpolatedVarSingleAxis
  j_toroidal_total: interpolated_param.InterpolatedVarSingleAxis
  j_toroidal_total_face: interpolated_param.InterpolatedVarSingleAxis
  delta_upper_face: interpolated_param.InterpolatedVarSingleAxis
  delta_lower_face: interpolated_param.InterpolatedVarSingleAxis
  elongation: interpolated_param.InterpolatedVarSingleAxis
  elongation_face: interpolated_param.InterpolatedVarSingleAxis
  connection_length_target: interpolated_param.InterpolatedVarSingleAxis | None
  connection_length_divertor: (
      interpolated_param.InterpolatedVarSingleAxis | None
  )
  angle_of_incidence_target: interpolated_param.InterpolatedVarSingleAxis | None
  R_OMP: interpolated_param.InterpolatedVarSingleAxis | None
  R_target: interpolated_param.InterpolatedVarSingleAxis | None
  B_pol_OMP: interpolated_param.InterpolatedVarSingleAxis | None
  diverted: interpolated_param.InterpolatedVarSingleAxis | None
  r0_vacuum: interpolated_param.InterpolatedVarSingleAxis | None

  def __call__(self, t: chex.Numeric) -> geometry.Geometry:
    """Returns a Geometry instance at the given time."""
    chex.assert_type(t, jnp.floating)
    return self._get_geometry_base(t, StandardGeometry)


@dataclasses.dataclass(frozen=True)
class StandardGeometryIntermediates:
  r"""Holds the intermediate values used to build a StandardGeometry.

  In particular these are the values that are used when interpolating different
  geometries.  These intermediates are typically extracted directly from
  equilibrium solver outputs (like CHEASE, MEQ, or EQDSK) and then used to
  construct a `StandardGeometry` instance.

  TODO(b/335204606): Specify the expected COCOS format.
  NOTE: Right now, TORAX does not have a specified COCOS format. Our team is
  working on adding this and updating documentation to make that clear. The
  CHEASE input data is still COCOS 2.

  All inputs are 1D profiles vs normalized rho toroidal (rhon).

  Attributes:
    geometry_type:  The type of geometry being represented (e.g., CHEASE, MEQ,
      EQDSK).
    R_major: Major radius defined as midplane average of the LCFS radius,
      :math:`R_\mathrm{major} = (R_\mathrm{max} + R_\mathrm{min}) / 2`
      [:math:`\mathrm{m}`].
    a_minor: Minor radius, defined as half the horizontal width of the LCFS,
      :math:`a_\mathrm{minor} = (R_\mathrm{max} - R_\mathrm{min}) / 2`
      [:math:`\mathrm{m}`].
    B_0: Vacuum toroidal magnetic field at ``R_major`` [:math:`\mathrm{T}`].
    psi: Poloidal flux profile [:math:`\mathrm{Wb}`].
    I_p_profile: Plasma current profile [:math:`\mathrm{A}`].
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
    trapped_fraction: Effective trapped particle fraction [dimensionless]; see
      `torax._src.geometry.trapped_fraction.TrappedFractionSource`.
    delta_upper_face: Upper triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    delta_lower_face: Lower triangularity [dimensionless]. See `Geometry`
      docstring for definition.
    elongation: Plasma elongation profile [dimensionless]. See `Geometry`
      docstring for definition.
    vpr:  Profile of dVolume/d(rho_norm), where rho_norm is the normalized
      toroidal flux coordinate [:math:`\mathrm{m^3}`].
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
      For a grid with N cells, there are N+1 faces.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations. Used to create a higher-resolution grid to improve accuracy
      when initializing psi from a plasma current profile.
    z_magnetic_axis: z position of magnetic axis [:math:`\mathrm{m}`].
    diverted: Boolean flag indicating whether the geometry is diverted.
    connection_length_target: Parallel connection length from outboard midplane
      to target [m].
    connection_length_divertor: Parallel connection length from outboard
      midplane to X-point [m].
    angle_of_incidence_target: Angle between magnetic field line and divertor
      target [degrees].
    R_OMP: Major radius of the outboard midplane [m].
    R_target: Major radius of the divertor target strike point [m].
    B_pol_OMP: Poloidal magnetic field at the outboard midplane [T].
    r0_vacuum: Reference major radius of the input vacuum toroidal field [m].
  """

  geometry_type: geometry.GeometryType
  R_major: array_typing.FloatScalar
  a_minor: array_typing.FloatScalar
  B_0: array_typing.FloatScalar
  psi: array_typing.Array
  I_p_profile: array_typing.Array
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
  face_centers: array_typing.Array
  hires_factor: int
  z_magnetic_axis: array_typing.FloatScalar | None
  # Optional parameters not present in all geometry sources.
  diverted: bool | None
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  angle_of_incidence_target: array_typing.FloatScalar | None
  R_OMP: array_typing.FloatScalar | None
  R_target: array_typing.FloatScalar | None
  B_pol_OMP: array_typing.FloatScalar | None
  trapped_fraction: array_typing.Array
  r0_vacuum: array_typing.FloatScalar | None = None

  def __post_init__(self):
    """Enforces sign conventions, extrapolates edge, and smooths near-axis.

    Sign conventions - TODO(b/335204606): Replace with proper COCOS handling:
    - psi must grow with radius, i.e. psi(axis) < psi(separatrix).
    - Total I_p must be positive.
    - Positive-definite quantities are enforced via abs(): Phi, F,
      int_dl_over_Bp, vpr, flux_surf_avg_grad_psi, flux_surf_avg_grad_psi2,
      flux_surf_avg_grad_psi2_over_R2.
    Then:
    - If the grid does not start on the magnetic axis, the axis is prepended.
    - Edge extrapolation for a subset of attributes based on a Cubic spline fit.
    - Near-axis smoothing for a subset of attributes based on an asymptotic
      polynomial fit in rho_norm anchored at the smoothing boundary.
    """

    if self.psi[-1] < self.psi[0]:
      object.__setattr__(self, 'psi', -self.psi)

    if self.I_p_profile[-1] < 0:
      object.__setattr__(self, 'I_p_profile', -self.I_p_profile)

    object.__setattr__(self, 'Phi', np.abs(self.Phi))
    object.__setattr__(self, 'F', np.abs(self.F))
    object.__setattr__(self, 'int_dl_over_Bp', np.abs(self.int_dl_over_Bp))
    object.__setattr__(self, 'vpr', np.abs(self.vpr))
    object.__setattr__(
        self, 'flux_surf_avg_grad_psi', np.abs(self.flux_surf_avg_grad_psi)
    )
    object.__setattr__(
        self, 'flux_surf_avg_grad_psi2', np.abs(self.flux_surf_avg_grad_psi2)
    )
    object.__setattr__(
        self,
        'flux_surf_avg_grad_psi2_over_R2',
        np.abs(self.flux_surf_avg_grad_psi2_over_R2),
    )

    # The geometry is built assuming that the first grid point is the magnetic
    # axis, but some equilibria do not start exactly at the magnetic axis. For
    # these cases, prepend rho_norm=0 to the relevant profiles.
    if self.Phi[0] > 0.0:
      self._prepend_magnetic_axis()

    rhon = np.sqrt(self.Phi / self.Phi[-1])

    # Check if last flux surface is diverted and correct via spline fit if so
    if self.diverted or self.flux_surf_avg_grad_psi2_over_R2[-1] < 1e-10:
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

    # Near-axis smoothing of quantities with known near-axis trends with rho.
    idx_limit = int(np.argmin(np.abs(rhon - _RHO_SMOOTHING_LIMIT)))

    # |grad(psi)| = R * Bp scales like rho_norm near the axis, so |grad(psi)|^2
    # terms scale as rho_norm^2 + O(rho_norm^4) and |grad(psi)| scales as
    # rho_norm + O(rho_norm^3).
    self.flux_surf_avg_grad_psi2_over_R2[:] = _smooth_near_axis(
        self.flux_surf_avg_grad_psi2_over_R2, rhon, idx_limit, order=2
    )
    self.flux_surf_avg_grad_psi2[:] = _smooth_near_axis(
        self.flux_surf_avg_grad_psi2, rhon, idx_limit, order=2
    )
    self.flux_surf_avg_grad_psi[:] = _smooth_near_axis(
        self.flux_surf_avg_grad_psi, rhon, idx_limit, order=1
    )
    # Warn if the input grid is too coarse, which can cause artifacts in
    # derived quantities such as j_toroidal_total.
    drho_norm = np.diff(rhon)
    max_spacing = np.max(drho_norm)
    if max_spacing > _COARSE_GRID_THRESHOLD:
      logging.log_every_n_seconds(
          logging.WARNING,
          'The input equilibrium grid is coarse (max rho_norm spacing ='
          ' %.4f > %.4f). This can create numerical artifacts in derived'
          ' quantities such as j_toroidal_total. Consider using a'
          ' higher-resolution equilibrium file.',
          _COARSE_GRID_WARNING_INTERVAL_SEC,  # log every interval
          max_spacing,
          _COARSE_GRID_THRESHOLD,
      )

  def _prepend_magnetic_axis(self) -> None:
    """Prepends the magnetic axis (rho_norm=0) to the radial profiles.

    Profiles that physically vanish on axis are set to zero there, and R_in and
    R_out are both set to the midplane center of the first flux surface. All
    other profiles are finite on axis and held at their value on the first flux
    surface.
    """
    R_axis = 0.5 * (np.asarray(self.R_in)[0] + np.asarray(self.R_out)[0])
    axis_values = dict.fromkeys(_ZERO_ON_AXIS, 0.0)
    axis_values['R_in'] = R_axis
    axis_values['R_out'] = R_axis
    for name, f in dataclasses.asdict(self).items():
      # Skip scalar attributes (ndim != 1) and `face_centers` (which is the
      # target TORAX mesh, not a profile from the input equilibrium).
      if name == 'face_centers' or np.ndim(f) != 1:
        continue
      f = np.asarray(f)
      object.__setattr__(
          self, name, np.insert(f, 0, axis_values.get(name, f[0]))
      )


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
  idx_limit = int(
      np.argmin(np.abs(rho_norm_intermediate - _RHO_SMOOTHING_LIMIT))
  )
  # dV/dpsi and 1/R^2 are finite on axis while |grad(psi)| scales like rho_norm,
  # so g0 (= <|grad(V)|>) scales as rho_norm (order=1), and g1 (= <|grad(V)|^2>)
  # and g2 (= <|grad(V)|^2 / R^2>) scale as rho_norm^2 (order=2).
  g0 = _smooth_near_axis(
      intermediates.flux_surf_avg_grad_psi * dV_dpsi,
      rho_norm_intermediate,
      idx_limit,
      order=1,
  )  # <\nabla V>
  g1 = _smooth_near_axis(
      intermediates.flux_surf_avg_grad_psi2 * dV_dpsi**2,
      rho_norm_intermediate,
      idx_limit,
      order=2,
  )  # <(\nabla V)^2>
  g2 = _smooth_near_axis(
      intermediates.flux_surf_avg_grad_psi2_over_R2 * dV_dpsi**2,
      rho_norm_intermediate,
      idx_limit,
      order=2,
  )  # <(\nabla V)^2 / R^2>
  g3 = intermediates.flux_surf_avg_1_over_R2  # <1/R**2>
  g2g3_over_rhon = g2[1:] * g3[1:] / rho_norm_intermediate[1:]
  g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))

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
  dI_tot_drhon = np.gradient(intermediates.I_p_profile, rho_norm_intermediate)

  j_toroidal_total_face_bulk = dI_tot_drhon[1:] / spr[1:]

  # For now set on-axis to the same as the second grid point, due to 0/0
  # division.
  j_toroidal_total_face_axis = j_toroidal_total_face_bulk[0]

  j_toroidal_total = np.concatenate(
      [np.array([j_toroidal_total_face_axis]), j_toroidal_total_face_bulk]
  )

  # fill geometry structure
  # normalized grid
  mesh = torax_pydantic.Grid1D(face_centers=intermediates.face_centers)
  rho_b = rho_intermediate[-1]  # radius denormalization constant
  # helper variables for mesh cells and faces
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current.
  rho_hires_norm = geometry.increase_grid_resolution(
      rho_face_norm, intermediates.hires_factor
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

  j_toroidal_total_face = rhon_interpolation_func(
      rho_face_norm, j_toroidal_total
  )
  j_toroidal_total = rhon_interpolation_func(rho_norm, j_toroidal_total)

  I_p_profile_face = rhon_interpolation_func(
      rho_face_norm, intermediates.I_p_profile
  )

  Rin_face = rhon_interpolation_func(rho_face_norm, intermediates.R_in)
  Rin = rhon_interpolation_func(rho_norm, intermediates.R_in)

  Rout_face = rhon_interpolation_func(rho_face_norm, intermediates.R_out)
  Rout = rhon_interpolation_func(rho_norm, intermediates.R_out)

  trapped_fraction_face = rhon_interpolation_func(
      rho_face_norm, intermediates.trapped_fraction
  )

  g0_face = rhon_interpolation_func(rho_face_norm, g0)
  g0 = rhon_interpolation_func(rho_norm, g0)

  # g1 and g2 scale like rho_norm^2 near the axis, while vpr scales like
  # rho_norm: flux surfaces near the axis are nested ellipses of minor radius r,
  # so both the volume and the toroidal flux scale like r^2, i.e. V ~ rho_norm^2
  # and vpr = dV/drho_norm ~ rho_norm, while |grad rho_norm| tends to a
  # constant. Linear interpolation within the first equilibrium interval would
  # make g1/vpr^2 and g2/vpr^2 diverge like 1/rho_norm there, so keep these
  # ratios constant in that interval instead.
  interp_near_axis = functools.partial(
      _interpolate_rho_norm_squared_near_axis,
      rho_norm_intermediate=rho_norm_intermediate,
      vpr_intermediate=intermediates.vpr,
  )
  g1_face = interp_near_axis(rho_face_norm, g1, vpr_face)
  g1 = interp_near_axis(rho_norm, g1, vpr)

  g2_face = interp_near_axis(rho_face_norm, g2, vpr_face)
  g2 = interp_near_axis(rho_norm, g2, vpr)

  g3_face = rhon_interpolation_func(rho_face_norm, g3)
  g3 = rhon_interpolation_func(rho_norm, g3)

  g2g3_over_rhon_face = rhon_interpolation_func(rho_face_norm, g2g3_over_rhon)
  g2g3_over_rhon_hires = rhon_interpolation_func(rho_hires_norm, g2g3_over_rhon)
  g2g3_over_rhon = rhon_interpolation_func(rho_norm, g2g3_over_rhon)

  # Alternative initial psi, consistent with the equilibrium I_p profile.
  psi_from_I_p, psi_from_I_p_face = _calculate_psi_from_I_p(
      rho_norm_intermediate=rho_norm_intermediate,
      I_p_profile=intermediates.I_p_profile,
      psi_axis=intermediates.psi[0],
      Phi_b=intermediates.Phi[-1],
      rho_norm=rho_norm,
      rho_face_norm=rho_face_norm,
      g2g3_over_rhon_face=g2g3_over_rhon_face,
      F_face=F_face,
  )

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
      trapped_fraction_face=trapped_fraction_face,
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
      I_p_profile_face=I_p_profile_face,
      psi=psi,
      psi_from_I_p=psi_from_I_p,
      psi_from_I_p_face=psi_from_I_p_face,
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
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
      angle_of_incidence_target=intermediates.angle_of_incidence_target,
      R_OMP=intermediates.R_OMP,
      R_target=intermediates.R_target,
      B_pol_OMP=intermediates.B_pol_OMP,
      r0_vacuum=intermediates.r0_vacuum,
  )


def _interpolate_rho_norm_squared_near_axis(
    rho_norm: array_typing.Array,
    y_intermediate: array_typing.Array,
    vpr: array_typing.Array,
    rho_norm_intermediate: array_typing.Array,
    vpr_intermediate: array_typing.Array,
) -> np.ndarray:
  """Linearly interpolates a quantity that scales like rho_norm^2 on axis.

  Within the first equilibrium interval, y / vpr^2 is held at its value at the
  first off-axis equilibrium point, since vpr scales like rho_norm there.

  Args:
    rho_norm: rho_norm values to interpolate to.
    y_intermediate: Quantity on the equilibrium grid.
    vpr: vpr at rho_norm.
    rho_norm_intermediate: rho_norm of the equilibrium grid.
    vpr_intermediate: vpr on the equilibrium grid.

  Returns:
    The quantity at rho_norm.
  """
  y = np.interp(rho_norm, rho_norm_intermediate, y_intermediate)
  first_interval = rho_norm < rho_norm_intermediate[1]
  y_over_vpr2 = y_intermediate[1] / vpr_intermediate[1] ** 2
  return np.where(first_interval, y_over_vpr2 * vpr**2, y)


def _calculate_psi_from_I_p(
    rho_norm_intermediate: array_typing.Array,
    I_p_profile: array_typing.Array,
    psi_axis: array_typing.FloatScalar,
    Phi_b: array_typing.FloatScalar,
    rho_norm: array_typing.Array,
    rho_face_norm: array_typing.Array,
    g2g3_over_rhon_face: array_typing.Array,
    F_face: array_typing.Array,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns psi on the cell and face grids, consistent with I_p_profile.

  I_p_profile is cubic-spline interpolated onto the transport faces, and the
  face psi gradients are obtained by inverting the finite-volume I_p relation
  used in `psi_calculations.calc_j_toroidal_total`. psi is then the cumulative
  sum of these gradients from the axis.

  Args:
    rho_norm_intermediate: Normalized toroidal flux coordinate of the
      equilibrium grid.
    I_p_profile: Enclosed plasma current on the equilibrium grid [A].
    psi_axis: Poloidal flux on the magnetic axis [Wb].
    Phi_b: Toroidal flux at the LCFS [Wb].
    rho_norm: Cell centers of the transport mesh.
    rho_face_norm: Face centers of the transport mesh.
    g2g3_over_rhon_face: g2g3_over_rhon on the transport faces.
    F_face: F on the transport faces.

  Returns:
    psi_from_I_p on the cell and face grids [Wb].
  """
  I_p_face = scipy.interpolate.CubicSpline(rho_norm_intermediate, I_p_profile)(
      rho_face_norm
  )
  dpsi_drhon_face = np.zeros_like(rho_face_norm)
  dpsi_drhon_face[1:] = (
      I_p_face[1:]
      * (16 * constants.CONSTANTS.mu_0 * np.pi**3 * Phi_b)
      / (g2g3_over_rhon_face[1:] * F_face[1:])
  )
  # dpsi/drho_norm is linear in rho_norm near the axis.
  psi_first_cell = (
      psi_axis + 0.5 * dpsi_drhon_face[1] * rho_norm[0] ** 2 / rho_face_norm[1]
  )
  psi = psi_first_cell + np.concatenate(
      [np.zeros(1), np.cumsum(dpsi_drhon_face[1:-1] * np.diff(rho_norm))]
  )
  psi_face = np.interp(rho_face_norm, rho_norm, psi)
  psi_face[0] = psi_axis
  psi_face[-1] = psi[-1] + dpsi_drhon_face[-1] * (
      rho_face_norm[-1] - rho_norm[-1]
  )
  return psi, psi_face


def _smooth_near_axis(
    data: array_typing.Array,
    rhon: array_typing.Array,
    idx_limit: int,
    order: int,
) -> np.ndarray:
  """Smooths near-axis data by fitting its asymptotic functional form in rhon.

  For a quantity y(rhon) with leading near-axis scaling rhon^p, dividing out
  rhon^p gives the asymptotic ratio y(rhon) / rhon^p, which is a smooth
  flux-surface profile (symmetric across the magnetic axis) with zero radial
  slope at rhon = 0. Its expansion around the axis therefore contains only even
  powers of rhon:

    y(rhon) = rhon^p * (c_1 + c_2 * rhon^2 + O(rhon^4))

  Enforcing C^0 continuity y(r_limit) = y_limit at
  (r_limit, y_limit) = (rhon[idx_limit], data[idx_limit]) eliminates
  c_1 = y_limit / r_limit^p - c_2 * r_limit^2, leaving a single free parameter
  c_2:

    y(r) = y_limit * (r / r_limit)^p + c_2 * r^p * (r^2 - r_limit^2)

  which is fit by unweighted least squares on the interior points
  0 <= k < idx_limit.

  Args:
    data: 1D array of data on the equilibrium grid (with rhon[0] == 0).
    rhon: 1D array of normalized toroidal flux coordinates.
    idx_limit: Index up to which the near-axis fit is applied.
    order: Leading power p of rhon on axis (1 or 2).

  Returns:
    Array with indices :idx_limit replaced by the anchored asymptotic fit.
    No-op if idx_limit <= 1.
  """
  smoothed = np.array(data, copy=True)
  if idx_limit <= 1:
    return smoothed
  r_limit = rhon[idx_limit]
  y_limit = smoothed[idx_limit]
  r_interior = np.asarray(rhon[:idx_limit])
  y_interior = smoothed[:idx_limit]
  leading_term = y_limit * (r_interior / r_limit) ** order
  if idx_limit <= 2:
    smoothed[:idx_limit] = leading_term
    return smoothed
  next_order_basis = (r_interior**order) * (r_interior**2 - r_limit**2)
  # Closed form for least-squares solution for c2.
  c2 = np.dot(next_order_basis, y_interior - leading_term) / np.dot(
      next_order_basis, next_order_basis
  )
  smoothed[:idx_limit] = leading_term + c2 * next_order_basis
  return smoothed
