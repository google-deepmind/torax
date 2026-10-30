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
from typing import Literal, TypeAlias, cast

from absl import logging
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import interpolated_param
from torax._src import jax_utils
from torax._src import math_utils
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

_RADIAL_PROFILES = (
    'psi',
    'I_p_profile',
    'Phi',
    'R_in',
    'R_out',
    'F',
    'int_dl_over_Bp',
    'flux_surf_avg_1_over_R',
    'flux_surf_avg_1_over_R2',
    'flux_surf_avg_grad_psi',
    'flux_surf_avg_grad_psi2',
    'flux_surf_avg_grad_psi2_over_R2',
    'flux_surf_avg_B2',
    'flux_surf_avg_1_over_B2',
    'delta_upper_face',
    'delta_lower_face',
    'elongation',
    'vpr',
    'trapped_fraction',
)

_SplineBcType: TypeAlias = Literal['natural', 'not-a-knot']


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
  """

  I_p_profile_face: array_typing.FloatVectorFace
  psi: array_typing.FloatVectorCell
  psi_from_I_p: array_typing.FloatVectorCell
  psi_from_I_p_face: array_typing.FloatVectorFace
  j_toroidal_total: array_typing.FloatVectorCell
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
    geometry_type: The type of geometry being represented (e.g., CHEASE, MEQ,
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
    vpr: Profile of dVolume/d(rho_norm), where rho_norm is the normalized
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
  diverted: array_typing.BoolScalar | None
  connection_length_target: array_typing.FloatScalar | None
  connection_length_divertor: array_typing.FloatScalar | None
  angle_of_incidence_target: array_typing.FloatScalar | None
  R_OMP: array_typing.FloatScalar | None
  R_target: array_typing.FloatScalar | None
  B_pol_OMP: array_typing.FloatScalar | None
  trapped_fraction: array_typing.Array

  def __post_init__(self):
    """Enforces sign conventions, extrapolates edge, and smooths near-axis."""
    # Prepending the magnetic axis changes the array length from N to N + 1
    # conditionally on the runtime value `Phi[0] > 0.0`, which is not supported
    # under `@jax.jit` (where output shapes must be static).
    # TODO(b/398191165): Move magnetic-axis prepending to equilibrium file or
    # config loading so this host-only non-JAX check can be removed.
    if not isinstance(self.Phi, jax.core.Tracer):
      if abs(float(np.asarray(self.Phi)[0])) > 0.0:
        self._prepend_magnetic_axis()

    object.__setattr__(
        self,
        'face_centers',
        np.asarray(self.face_centers, dtype=jax_utils.get_np_dtype()),
    )

    dtype = jax_utils.get_dtype()
    profiles = {
        k: jnp.asarray(getattr(self, k), dtype=dtype) for k in _RADIAL_PROFILES
    }
    diverted_override = jnp.asarray(
        False if self.diverted is None else self.diverted, dtype=bool
    )
    rhon, processed = _preprocess_intermediate_profiles(
        profiles, diverted_override
    )
    for k, v in processed.items():
      object.__setattr__(self, k, v)
    _warn_if_coarse_grid(rhon)

  def _prepend_magnetic_axis(self) -> None:
    """Prepends the magnetic axis (rho_norm=0) to the radial profiles."""
    R_axis = 0.5 * (np.asarray(self.R_in)[0] + np.asarray(self.R_out)[0])
    axis_values = dict.fromkeys(_ZERO_ON_AXIS, 0.0)
    axis_values['R_in'] = R_axis
    axis_values['R_out'] = R_axis
    for name, f in vars(self).items():
      if name == 'face_centers' or np.ndim(f) != 1:
        continue
      axis_value = axis_values.get(name, np.asarray(f)[0])
      object.__setattr__(self, name, np.insert(f, 0, axis_value))


@jax.jit
def _preprocess_intermediate_profiles(
    profiles: dict[str, jax.Array],
    diverted_override: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Preprocesses equilibrium profiles: signs, edge extrapolation, smoothing."""
  p = dict(profiles)
  # Sign conventions - TODO(b/335204606): Replace with proper COCOS handling:
  # - psi must grow with radius, i.e. psi(axis) < psi(separatrix).
  # - Total I_p must be positive.
  # - Positive-definite quantities are enforced via abs().
  p['psi'] = jnp.where(p['psi'][-1] < p['psi'][0], -p['psi'], p['psi'])
  p['I_p_profile'] = jnp.where(
      p['I_p_profile'][-1] < 0, -p['I_p_profile'], p['I_p_profile']
  )
  for k in (
      'Phi',
      'F',
      'int_dl_over_Bp',
      'vpr',
      'flux_surf_avg_grad_psi',
      'flux_surf_avg_grad_psi2',
      'flux_surf_avg_grad_psi2_over_R2',
  ):
    p[k] = jnp.abs(p[k])

  rhon = math_utils.sqrt_with_zero_gradient_at_zero(p['Phi'] / p['Phi'][-1])

  # Diverted edge extrapolation:
  # Quantities diverging at an X-point have singular values at the plasma
  # boundary. If diverted, extrapolate edge values with cubic splines.
  is_diverted = (
      p['flux_surf_avg_grad_psi2_over_R2'][-1] < 1e-10
  ) | diverted_override

  def _extrapolate_when_diverted() -> dict[str, jax.Array]:
    edge_grad_psi2 = _cubic_spline_extrapolate_edge(
        rhon, p['flux_surf_avg_grad_psi2_over_R2'], 1.0, 'not-a-knot'
    )
    edge_int_dl = _cubic_spline_extrapolate_edge(
        rhon, p['int_dl_over_Bp'], 1.0, 'not-a-knot'
    )
    g2_edge_ratio = (edge_grad_psi2 * edge_int_dl**2) / (
        p['flux_surf_avg_grad_psi2_over_R2'][-2] * p['int_dl_over_Bp'][-2] ** 2
    )
    bc_type_is_not_a_knot = g2_edge_ratio > 1.0

    def _extrapolate_with_bc(
        bc_type: _SplineBcType, e_grad2: jax.Array, e_dl: jax.Array
    ) -> dict[str, jax.Array]:
      res = dict(p)
      res['flux_surf_avg_grad_psi2_over_R2'] = _with_replaced_edge(
          res['flux_surf_avg_grad_psi2_over_R2'], e_grad2
      )
      res['int_dl_over_Bp'] = _with_replaced_edge(res['int_dl_over_Bp'], e_dl)
      for name in (
          'flux_surf_avg_1_over_R2',
          'flux_surf_avg_grad_psi',
          'flux_surf_avg_grad_psi2',
          'vpr',
      ):
        e_val = _cubic_spline_extrapolate_edge(rhon, res[name], 1.0, bc_type)
        res[name] = _with_replaced_edge(res[name], e_val)
      return res

    return jax.lax.cond(
        bc_type_is_not_a_knot,
        lambda: _extrapolate_with_bc(
            'not-a-knot', edge_grad_psi2, edge_int_dl
        ),
        lambda: _extrapolate_with_bc(
            'natural',
            _cubic_spline_extrapolate_edge(
                rhon, p['flux_surf_avg_grad_psi2_over_R2'], 1.0, 'natural'
            ),
            _cubic_spline_extrapolate_edge(
                rhon, p['int_dl_over_Bp'], 1.0, 'natural'
            ),
        ),
    )

  extrapolated = jax.lax.cond(
      is_diverted,
      _extrapolate_when_diverted,
      lambda: p,
  )

  # Near-axis smoothing:
  # Equilibrium solvers approximate profiles near the magnetic axis with
  # finite-resolution polynomials that can suffer from non-physical radial slope
  # artifacts. Enforce correct asymptotic scaling near axis.
  idx_limit = jnp.argmin(jnp.abs(rhon - _RHO_SMOOTHING_LIMIT))
  smoothed = dict(extrapolated)
  smoothed['flux_surf_avg_grad_psi2_over_R2'] = _smooth_near_axis(
      smoothed['flux_surf_avg_grad_psi2_over_R2'], rhon, idx_limit, order=2
  )
  smoothed['flux_surf_avg_grad_psi2'] = _smooth_near_axis(
      smoothed['flux_surf_avg_grad_psi2'], rhon, idx_limit, order=2
  )
  smoothed['flux_surf_avg_grad_psi'] = _smooth_near_axis(
      smoothed['flux_surf_avg_grad_psi'], rhon, idx_limit, order=1
  )

  return rhon, smoothed


@functools.partial(jax.jit, static_argnames=('hires_factor',))
def _build_standard_geometry_jitted(
    profiles: dict[str, jax.Array],
    mesh: torax_pydantic.Grid1D,
    hires_factor: int,
) -> StandardGeometry:
  """Jitted constructor for standard geometry."""
  dtype = jax_utils.get_dtype()
  Phi = profiles['Phi']
  B_0 = profiles['B_0']
  R_major = profiles['R_major']
  a_minor = profiles['a_minor']

  # Define toroidal flux coordinate (rho) at boundary (LCFS)
  # rho = sqrt(Phi / (pi * B_0))
  # derived from Phi = pi * r^2 * B_0
  rho_intermediate = math_utils.sqrt_with_zero_gradient_at_zero(
      Phi / (jnp.pi * B_0)
  )
  rho_b = rho_intermediate[-1]
  rho_norm_intermediate = rho_intermediate / rho_b

  # Interpolate the profiles from the intermediate grid to the face and cell
  # grids of the simulation.
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers

  # Function to interpolate profiles to the cell and face grids.
  interp = lambda x, y: jnp.interp(x, rho_norm_intermediate, y)

  # Define the grid profiles.
  Phi_interp = interp(rho_norm, Phi)
  Phi_face = interp(rho_face_norm, Phi)
  R_in = interp(rho_norm, profiles['R_in'])
  R_in_face = interp(rho_face_norm, profiles['R_in'])
  R_out = interp(rho_norm, profiles['R_out'])
  R_out_face = interp(rho_face_norm, profiles['R_out'])
  F = interp(rho_norm, profiles['F'])
  F_face = interp(rho_face_norm, profiles['F'])

  # dV/dpsi = int_dl_over_Bp
  dV_dpsi = profiles['int_dl_over_Bp']
  idx_limit = jnp.argmin(jnp.abs(rho_norm_intermediate - _RHO_SMOOTHING_LIMIT))
  g0_intermediate = _smooth_near_axis(
      profiles['flux_surf_avg_grad_psi'] * dV_dpsi,
      rho_norm_intermediate,
      idx_limit,
      order=1,
  )
  g1_intermediate = _smooth_near_axis(
      profiles['flux_surf_avg_grad_psi2'] * dV_dpsi**2,
      rho_norm_intermediate,
      idx_limit,
      order=2,
  )
  g2_intermediate = _smooth_near_axis(
      profiles['flux_surf_avg_grad_psi2_over_R2'] * dV_dpsi**2,
      rho_norm_intermediate,
      idx_limit,
      order=2,
  )
  g3_intermediate = profiles['flux_surf_avg_1_over_R2']
  g2g3_over_rhon_intermediate = jnp.concatenate([
      jnp.zeros(1),
      g2_intermediate[1:] * g3_intermediate[1:] / rho_norm_intermediate[1:],
  ])

  # Intermediate profiles for volume and area are defined by numerically
  # integrating vpr and spr on the intermediate grid.
  vpr_intermediate = profiles['vpr']
  spr_intermediate = (
      vpr_intermediate * profiles['flux_surf_avg_1_over_R'] / (2 * jnp.pi)
  )
  volume_intermediate = math_utils.cumulative_trapezoid(
      y=vpr_intermediate, x=rho_norm_intermediate, initial=0.0
  )
  area_intermediate = math_utils.cumulative_trapezoid(
      y=spr_intermediate, x=rho_norm_intermediate, initial=0.0
  )

  vpr = interp(rho_norm, vpr_intermediate)
  vpr_face = interp(rho_face_norm, vpr_intermediate)
  spr = interp(rho_norm, spr_intermediate)
  spr_face = interp(rho_face_norm, spr_intermediate)
  volume = interp(rho_norm, volume_intermediate)
  volume_face = interp(rho_face_norm, volume_intermediate)
  area = interp(rho_norm, area_intermediate)
  area_face = interp(rho_face_norm, area_intermediate)

  # delta_face is the average of delta_upper_face and delta_lower_face.
  delta_upper_face = interp(rho_face_norm, profiles['delta_upper_face'])
  delta_lower_face = interp(rho_face_norm, profiles['delta_lower_face'])
  delta_face = 0.5 * (delta_upper_face + delta_lower_face)
  elongation = interp(rho_norm, profiles['elongation'])
  elongation_face = interp(rho_face_norm, profiles['elongation'])
  trapped_fraction_face = interp(rho_face_norm, profiles['trapped_fraction'])

  g0 = interp(rho_norm, g0_intermediate)
  g0_face = interp(rho_face_norm, g0_intermediate)

  # Near axis g1 and g2 are set to be proportional to rho_norm^2 to avoid
  # numerical noise in the input profiles.
  interp_near_axis = functools.partial(
      _interpolate_rho_norm_squared_near_axis,
      rho_norm_intermediate=rho_norm_intermediate,
      vpr_intermediate=vpr_intermediate,
  )
  g1 = interp_near_axis(rho_norm, g1_intermediate, vpr)
  g1_face = interp_near_axis(rho_face_norm, g1_intermediate, vpr_face)
  g2 = interp_near_axis(rho_norm, g2_intermediate, vpr)
  g2_face = interp_near_axis(rho_face_norm, g2_intermediate, vpr_face)
  g3 = interp(rho_norm, g3_intermediate)
  g3_face = interp(rho_face_norm, g3_intermediate)
  g2g3_over_rhon = interp(rho_norm, g2g3_over_rhon_intermediate)
  g2g3_over_rhon_face = interp(rho_face_norm, g2g3_over_rhon_intermediate)

  gm4 = interp(rho_norm, profiles['flux_surf_avg_1_over_B2'])
  gm4_face = interp(rho_face_norm, profiles['flux_surf_avg_1_over_B2'])
  gm5 = interp(rho_norm, profiles['flux_surf_avg_B2'])
  gm5_face = interp(rho_face_norm, profiles['flux_surf_avg_B2'])

  # Total toroidal current density
  dI_tot_drhon = cast(
      jax.Array, jnp.gradient(profiles['I_p_profile'], rho_norm_intermediate)
  )
  j_toroidal_total_face_bulk = dI_tot_drhon[1:] / spr_intermediate[1:]
  j_toroidal_total_intermediate = jnp.concatenate(
      [j_toroidal_total_face_bulk[:1], j_toroidal_total_face_bulk]
  )
  j_toroidal_total = interp(rho_norm, j_toroidal_total_intermediate)
  j_toroidal_total_face = interp(rho_face_norm, j_toroidal_total_intermediate)
  I_p_profile_face = interp(rho_face_norm, profiles['I_p_profile'])

  psi = interp(rho_norm, profiles['psi'])
  psi_from_I_p, psi_from_I_p_face = _calculate_psi_from_I_p(
      rho_norm_intermediate=rho_norm_intermediate,
      I_p_profile=profiles['I_p_profile'],
      psi_axis=profiles['psi'][0],
      Phi_b=Phi[-1],
      rho_norm=rho_norm,
      rho_face_norm=rho_face_norm,
      g2g3_over_rhon_face=g2g3_over_rhon_face,
      F_face=F_face,
  )

  # High-resolution grid profiles
  rho_hires_norm = geometry.increase_grid_resolution(
      rho_face_norm, hires_factor
  )
  rho_hires = rho_hires_norm * rho_b
  g2g3_over_rhon_hires = interp(rho_hires_norm, g2g3_over_rhon_intermediate)
  F_hires = interp(rho_hires_norm, profiles['F'])
  spr_hires = interp(rho_hires_norm, spr_intermediate)

  return StandardGeometry(
      geometry_type=geometry.GeometryType.CHEASE,
      torax_mesh=mesh,
      Phi=Phi_interp,
      Phi_face=Phi_face,
      R_major=R_major,
      a_minor=a_minor,
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
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
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
      elongation=elongation,
      elongation_face=elongation_face,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      Phi_b_dot=jnp.zeros((), dtype=dtype),
      _z_magnetic_axis=jnp.asarray(0.0, dtype=dtype),
      I_p_profile_face=I_p_profile_face,
      psi=psi,
      psi_from_I_p=psi_from_I_p,
      psi_from_I_p_face=psi_from_I_p_face,
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
      connection_length_target=None,
      connection_length_divertor=None,
      angle_of_incidence_target=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
      diverted=None,
  )


@array_typing.jaxtyped
def build_standard_geometry(
    intermediates: StandardGeometryIntermediates,
) -> StandardGeometry:
  """Build geometry object based on set of profiles from an EQ solution."""
  mesh = torax_pydantic.Grid1D(
      face_centers=np.asarray(intermediates.face_centers)
  )
  dtype = jax_utils.get_dtype()
  profiles = {
      k: jnp.asarray(getattr(intermediates, k), dtype=dtype)
      for k in _RADIAL_PROFILES
  }
  profiles['R_major'] = jnp.asarray(intermediates.R_major, dtype=dtype)
  profiles['a_minor'] = jnp.asarray(intermediates.a_minor, dtype=dtype)
  profiles['B_0'] = jnp.asarray(intermediates.B_0, dtype=dtype)

  geo = _build_standard_geometry_jitted(
      profiles=profiles,
      mesh=mesh,
      hires_factor=intermediates.hires_factor,
  )
  return dataclasses.replace(
      geo,
      geometry_type=intermediates.geometry_type,
      torax_mesh=mesh,
      _z_magnetic_axis=intermediates.z_magnetic_axis,
      diverted=intermediates.diverted,
      connection_length_target=intermediates.connection_length_target,
      connection_length_divertor=intermediates.connection_length_divertor,
      angle_of_incidence_target=intermediates.angle_of_incidence_target,
      R_OMP=intermediates.R_OMP,
      R_target=intermediates.R_target,
      B_pol_OMP=intermediates.B_pol_OMP,
  )


def _solve_cubic_spline_moments(
    x: jax.Array,
    h: jax.Array,
    delta: jax.Array,
    bc_type: _SplineBcType,
) -> jax.Array:
  """Solves for cubic spline derivatives at the knots."""
  dl_int = h[1:]
  d_int = 2.0 * (h[:-1] + h[1:])
  du_int = h[:-1]
  rhs_int = 3.0 * (h[1:] * delta[:-1] + h[:-1] * delta[1:])
  if bc_type == 'natural':
    d0 = jnp.array(2.0, dtype=x.dtype)
    du0 = jnp.array(1.0, dtype=x.dtype)
    r0 = 3.0 * delta[0]
    dln = jnp.array(1.0, dtype=x.dtype)
    dn = jnp.array(2.0, dtype=x.dtype)
    rn = 3.0 * delta[-1]
  elif bc_type == 'not-a-knot':
    d_left = x[2] - x[0]
    d0 = h[1]
    du0 = d_left
    r0 = (
        (h[0] + 2.0 * d_left) * h[1] * delta[0] + h[0] ** 2 * delta[1]
    ) / d_left
    d_right = x[-1] - x[-3]
    dln = d_right
    dn = h[-2]
    rn = (
        h[-1] ** 2 * delta[-2] + (2.0 * d_right + h[-1]) * h[-2] * delta[-1]
    ) / d_right
  else:
    raise ValueError(f'Unsupported bc_type: {bc_type}')
  dl = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), dl_int, dln[None]])
  d = jnp.concatenate([d0[None], d_int, dn[None]])
  du = jnp.concatenate([du0[None], du_int, jnp.zeros(1, dtype=x.dtype)])
  rhs = jnp.concatenate([r0[None], rhs_int, rn[None]])[:, None]
  return jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs)[:, 0]


def _cubic_spline_eval(
    x: array_typing.Array,
    y: array_typing.Array,
    x_eval: array_typing.Array | float,
    bc_type: _SplineBcType,
) -> jax.Array:
  """Evaluates a 1D cubic spline at x_eval using pure JAX."""
  x_arr = jnp.asarray(x)
  y_arr = jnp.asarray(y, dtype=jnp.result_type(x_arr, y))
  xq = jnp.asarray(x_eval, dtype=x_arr.dtype)
  h = jnp.diff(x_arr)
  delta = jnp.diff(y_arr) / h
  s = _solve_cubic_spline_moments(x_arr, h, delta, bc_type)
  idx = jnp.clip(
      jnp.searchsorted(x_arr, xq, side='right') - 1, 0, x_arr.shape[0] - 2
  )
  dx = xq - x_arr[idx]
  h_i = h[idx]
  delta_i = delta[idx]
  s_i = s[idx]
  s_ip1 = s[idx + 1]
  c2 = (3.0 * delta_i - 2.0 * s_i - s_ip1) / h_i
  c3 = (s_i + s_ip1 - 2.0 * delta_i) / (h_i**2)
  return y_arr[idx] + dx * (s_i + dx * (c2 + dx * c3))


def _cubic_spline_extrapolate_edge(
    rho: array_typing.Array,
    data: array_typing.Array,
    x: float | jax.Array,
    bc_type: _SplineBcType,
) -> jax.Array:
  """Extrapolates edge using 1D cubic spline evaluated at x."""
  return _cubic_spline_eval(rho[:-1], data[:-1], x, bc_type)


def _with_replaced_edge(
    array: array_typing.Array,
    edge_value: chex.Numeric,
) -> jax.Array:
  """Returns a new array with the last element replaced by edge_value."""
  array = jnp.asarray(array)
  edge_array = jnp.reshape(
      jnp.asarray(edge_value, dtype=array.dtype),
      (1,),
  )
  return jnp.concatenate([array[:-1], edge_array])


def _interpolate_rho_norm_squared_near_axis(
    rho_norm: array_typing.Array,
    y_intermediate: array_typing.Array,
    vpr: array_typing.Array,
    rho_norm_intermediate: array_typing.Array,
    vpr_intermediate: array_typing.Array,
) -> jax.Array:
  """Linearly interpolates a quantity that scales like rho_norm^2 on axis."""
  rho_norm = jnp.asarray(rho_norm)
  rho_norm_intermediate = jnp.asarray(rho_norm_intermediate)
  y_intermediate = jnp.asarray(y_intermediate)
  vpr = jnp.asarray(vpr)
  vpr_intermediate = jnp.asarray(vpr_intermediate)
  y = jnp.interp(rho_norm, rho_norm_intermediate, y_intermediate)
  first_interval = rho_norm < rho_norm_intermediate[1]
  safe_vpr1 = jnp.where(vpr_intermediate[1] != 0.0, vpr_intermediate[1], 1.0)
  y_over_vpr2 = y_intermediate[1] / safe_vpr1**2
  return jnp.where(first_interval, y_over_vpr2 * vpr**2, y)


def _calculate_psi_from_I_p(
    rho_norm_intermediate: array_typing.Array,
    I_p_profile: array_typing.Array,
    psi_axis: array_typing.FloatScalar,
    Phi_b: array_typing.FloatScalar,
    rho_norm: array_typing.Array,
    rho_face_norm: array_typing.Array,
    g2g3_over_rhon_face: jax.Array,
    F_face: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Returns psi on the cell and face grids, consistent with I_p_profile."""
  I_p_face = _cubic_spline_eval(
      rho_norm_intermediate, I_p_profile, rho_face_norm, bc_type='not-a-knot'
  )
  dpsi_drhon_face = jnp.concatenate([
      jnp.zeros(1, dtype=I_p_face.dtype),
      I_p_face[1:]
      * (16 * constants.CONSTANTS.mu_0 * jnp.pi**3 * Phi_b)
      / (g2g3_over_rhon_face[1:] * F_face[1:]),
  ])
  psi_first_cell = (
      psi_axis + 0.5 * dpsi_drhon_face[1] * rho_norm[0] ** 2 / rho_face_norm[1]
  )
  psi = psi_first_cell + jnp.concatenate(
      [jnp.zeros(1), jnp.cumsum(dpsi_drhon_face[1:-1] * jnp.diff(rho_norm))]
  )
  psi_face_interp = jnp.interp(rho_face_norm, rho_norm, psi)
  edge_psi_face = psi[-1] + dpsi_drhon_face[-1] * (
      rho_face_norm[-1] - rho_norm[-1]
  )
  psi_face = jnp.concatenate([
      jnp.asarray([psi_axis]),
      psi_face_interp[1:-1],
      jnp.asarray([edge_psi_face]),
  ])
  return cast(jax.Array, psi), psi_face


def _smooth_near_axis(
    data: array_typing.Array,
    rhon: array_typing.Array,
    idx_limit: array_typing.IntScalar,
    order: int,
) -> jax.Array:
  """Smooths near-axis data by fitting its asymptotic functional form in rhon."""
  data_arr = jnp.asarray(data)
  rhon_arr = jnp.asarray(rhon)
  interior_mask = jnp.arange(data_arr.shape[0]) < idx_limit
  r_limit = rhon_arr[idx_limit]
  y_limit = data_arr[idx_limit]
  safe_r_limit = jnp.where(r_limit > 0.0, r_limit, 1.0)
  leading_term = jnp.where(
      r_limit > 0.0, y_limit * (rhon_arr / safe_r_limit) ** order, data_arr
  )
  next_order_basis = jnp.where(
      interior_mask,
      (rhon_arr**order) * (rhon_arr**2 - r_limit**2),
      0.0,
  )
  residual = jnp.where(interior_mask, data_arr - leading_term, 0.0)
  denom = jnp.dot(next_order_basis, next_order_basis)
  safe_denom = jnp.where(denom > 0.0, denom, 1.0)
  c2 = jnp.where(
      (idx_limit > 2) & (denom > 0.0),
      jnp.dot(next_order_basis, residual) / safe_denom,
      0.0,
  )
  fitted = leading_term + c2 * next_order_basis
  return jnp.where((idx_limit > 1) & interior_mask, fitted, data_arr)


def _warn_if_coarse_grid(rhon: jax.Array | np.ndarray) -> None:
  """Warns if the equilibrium grid is too coarse, risking artifacts in j_toroidal_total."""
  if isinstance(rhon, jax.core.Tracer):
    return
  drho_norm = np.diff(np.asarray(rhon))
  max_spacing = np.max(drho_norm)
  if max_spacing > _COARSE_GRID_THRESHOLD:
    logging.log_every_n_seconds(
        logging.WARNING,
        'The input equilibrium grid is coarse (max rho_norm spacing ='
        ' %.4f > %.4f). This can create numerical artifacts in derived'
        ' quantities such as j_toroidal_total. Consider using a'
        ' higher-resolution equilibrium file.',
        _COARSE_GRID_WARNING_INTERVAL_SEC,
        max_spacing,
        _COARSE_GRID_THRESHOLD,
    )
