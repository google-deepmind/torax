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

"""Classes for representing the problem geometry."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import functools
from typing import Type

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from torax import constants
from torax import geometry_loader
from torax import interpolated_param
from torax import jax_utils


@chex.dataclass(frozen=True)
class Grid1D:
  """Data structure defining a 1-D grid of cells with faces.

  Construct via `construct` classmethod.

  Attributes:
    nx: Number of cells.
    dx: Distance between cell centers.
    face_centers: Coordinates of face centers.
    cell_centers: Coordinates of cell centers.
  """

  nx: int
  dx: float
  face_centers: chex.Array
  cell_centers: chex.Array

  def __post_init__(self):
    jax_utils.assert_rank(self.nx, 0)
    jax_utils.assert_rank(self.dx, 0)
    jax_utils.assert_rank(self.face_centers, 1)
    jax_utils.assert_rank(self.cell_centers, 1)

  def __eq__(self, other: Grid1D) -> bool:
    return (
        self.nx == other.nx
        and self.dx == other.dx
        and np.array_equal(self.face_centers, other.face_centers)
        and np.array_equal(self.cell_centers, other.cell_centers)
    )

  def __hash__(self) -> int:
    return hash((self.nx, self.dx))

  @classmethod
  def construct(cls, nx: int, dx: float) -> Grid1D:
    """Constructs a Grid1D.

    Args:
      nx: Number of cells.
      dx: Distance between cell centers.

    Returns:
      grid: A Grid1D with the remaining fields filled in.
    """

    # Note: nx needs to be an int so that the shape `nx + 1` is not a Jax
    # tracer.

    return Grid1D(
        nx=nx,
        dx=dx,
        face_centers=np.linspace(0, nx * dx, nx + 1),
        cell_centers=np.linspace(dx * 0.5, (nx - 0.5) * dx, nx),
    )


def face_to_cell(face: chex.Array) -> chex.Array:
  """Infers cell values corresponding to a vector of face values.

  Simply a linear interpolation between face values.

  Args:
    face: An array containing face values.

  Returns:
    cell: An array containing cell values.
  """

  return 0.5 * (face[:-1] + face[1:])


@enum.unique
class GeometryType(enum.Enum):
  """Integer enum for geometry type.

  This type can be used within JAX expressions to access the geometry type
  without having to call isinstance.
  """

  CIRCULAR = 0
  CHEASE = 1


# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class Geometry:
  """Describes the magnetic geometry.

  Most users should default to using the StandardGeometry class, whether the
  source of their geometry comes from CHEASE, MEQ, etc.
  """

  # TODO(b/356356966): extend documentation to define what each attribute is.
  geometry_type: int
  torax_mesh: Grid1D
  dr_norm: chex.Array
  rmax: chex.Array
  Rmaj: chex.Array
  Rmin: chex.Array
  B0: chex.Array
  volume: chex.Array
  volume_face: chex.Array
  area: chex.Array
  area_face: chex.Array
  vpr: chex.Array
  vpr_face: chex.Array
  spr_cell: chex.Array
  spr_face: chex.Array
  delta_face: chex.Array
  g0: chex.Array
  g0_face: chex.Array
  g1: chex.Array
  g1_face: chex.Array
  g2: chex.Array
  g2_face: chex.Array
  g3: chex.Array
  g3_face: chex.Array
  g2g3_over_rhon: chex.Array
  g2g3_over_rhon_face: chex.Array
  g2g3_over_rhon_hires: chex.Array
  F: chex.Array
  F_face: chex.Array
  F_hires: chex.Array
  Rin: chex.Array
  Rin_face: chex.Array
  Rout: chex.Array
  Rout_face: chex.Array
  volume_hires: chex.Array
  area_hires: chex.Array
  spr_hires: chex.Array
  r_hires_norm: chex.Array
  r_hires: chex.Array
  vpr_hires: chex.Array
  Phibdot: chex.Array

  @property
  def r_norm(self) -> chex.Array:
    return self.torax_mesh.cell_centers

  @property
  def r_face_norm(self) -> chex.Array:
    return self.torax_mesh.face_centers

  @property
  def r_face(self) -> chex.Array:
    return self.r_face_norm * self.rmax

  @property
  def r(self) -> chex.Array:
    return self.r_norm * self.rmax

  @property
  def dr(self) -> chex.Array:
    return self.dr_norm * self.rmax

  # Toroidal flux at boundary (LCFS)
  @property
  def Phib(self) -> chex.Array:
    return self.rmax**2 * np.pi * self.B0

  @property
  def g1_over_vpr(self) -> chex.Array:
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1) / self.rmax,  # correct value is 1/rmax on-axis
        self.g0_face[1:] / self.vpr_face[1:],  # avoid div by zero on-axis
    ))

  @property
  def g1_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.zeros(1),  # correct value is zero on-axis
        self.g1_face[1:] / self.vpr_face[1:],  # avoid div by zero on-axis
    ))

  @property
  def g1_over_vpr2_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1) / self.rmax**2,  # correct value is 1/rmax**2 on-axis
        self.g1_face[1:] / self.vpr_face[1:] ** 2,  # avoid div by zero on-axis
    ))


@chex.dataclass(frozen=True)
class GeometryProvider:
  """A geometry which holds variables to interpolated based on time."""

  geometry_type: int
  torax_mesh: Grid1D
  dr_norm: interpolated_param.InterpolatedVarSingleAxis
  rmax: interpolated_param.InterpolatedVarSingleAxis
  Rmaj: interpolated_param.InterpolatedVarSingleAxis
  Rmin: interpolated_param.InterpolatedVarSingleAxis
  B0: interpolated_param.InterpolatedVarSingleAxis
  volume: interpolated_param.InterpolatedVarSingleAxis
  volume_face: interpolated_param.InterpolatedVarSingleAxis
  area: interpolated_param.InterpolatedVarSingleAxis
  area_face: interpolated_param.InterpolatedVarSingleAxis
  vpr: interpolated_param.InterpolatedVarSingleAxis
  vpr_face: interpolated_param.InterpolatedVarSingleAxis
  spr_cell: interpolated_param.InterpolatedVarSingleAxis
  spr_face: interpolated_param.InterpolatedVarSingleAxis
  delta_face: interpolated_param.InterpolatedVarSingleAxis
  g0: interpolated_param.InterpolatedVarSingleAxis
  g0_face: interpolated_param.InterpolatedVarSingleAxis
  g1: interpolated_param.InterpolatedVarSingleAxis
  g1_face: interpolated_param.InterpolatedVarSingleAxis
  g2: interpolated_param.InterpolatedVarSingleAxis
  g2_face: interpolated_param.InterpolatedVarSingleAxis
  g3: interpolated_param.InterpolatedVarSingleAxis
  g3_face: interpolated_param.InterpolatedVarSingleAxis
  g2g3_over_rhon: interpolated_param.InterpolatedVarSingleAxis
  g2g3_over_rhon_face: interpolated_param.InterpolatedVarSingleAxis
  g2g3_over_rhon_hires: interpolated_param.InterpolatedVarSingleAxis
  F: interpolated_param.InterpolatedVarSingleAxis
  F_face: interpolated_param.InterpolatedVarSingleAxis
  F_hires: interpolated_param.InterpolatedVarSingleAxis
  Rin: interpolated_param.InterpolatedVarSingleAxis
  Rin_face: interpolated_param.InterpolatedVarSingleAxis
  Rout: interpolated_param.InterpolatedVarSingleAxis
  Rout_face: interpolated_param.InterpolatedVarSingleAxis
  volume_hires: interpolated_param.InterpolatedVarSingleAxis
  area_hires: interpolated_param.InterpolatedVarSingleAxis
  spr_hires: interpolated_param.InterpolatedVarSingleAxis
  r_hires_norm: interpolated_param.InterpolatedVarSingleAxis
  r_hires: interpolated_param.InterpolatedVarSingleAxis
  vpr_hires: interpolated_param.InterpolatedVarSingleAxis

  @classmethod
  def create_provider(
      cls, geometries: Mapping[float, Geometry]
  ) -> GeometryProvider:
    """Creates a GeometryProvider from a mapping of times to geometries."""
    # Create a list of times and geometries.
    times = np.asarray(list(geometries.keys()))
    geos = list(geometries.values())
    initial_geometry = geos[0]
    for geometry in geos:
      if geometry.geometry_type != initial_geometry.geometry_type:
        raise ValueError('All geometries must have the same geometry type.')
      if geometry.torax_mesh != initial_geometry.torax_mesh:
        raise ValueError('All geometries must have the same mesh.')
    # Create a list of interpolated parameters for each geometry attribute.
    kwargs = {
        'geometry_type': initial_geometry.geometry_type,
        'torax_mesh': initial_geometry.torax_mesh,
    }
    if hasattr(initial_geometry, 'Ip_from_parameters'):
      kwargs['Ip_from_parameters'] = initial_geometry.Ip_from_parameters
    for attr in dataclasses.fields(cls):
      if (
          attr.name == 'geometry_type'
          or attr.name == 'torax_mesh'
          or attr.name == 'Ip_from_parameters'
      ):
        continue
      kwargs[attr.name] = interpolated_param.InterpolatedVarSingleAxis(
          (times, np.stack([getattr(g, attr.name) for g in geos], axis=0))
      )
    return cls(**kwargs)

  def _get_geometry_base(self, t: chex.Numeric, geometry_class: Type[Geometry]):
    """Returns a Geometry instance of the given type at the given time."""
    kwargs = {
        'geometry_type': self.geometry_type,
        'torax_mesh': self.torax_mesh,
    }
    if hasattr(self, 'Ip_from_parameters'):
      kwargs['Ip_from_parameters'] = self.Ip_from_parameters
    for attr in dataclasses.fields(geometry_class):
      if (
          attr.name == 'geometry_type'
          or attr.name == 'torax_mesh'
          or attr.name == 'Ip_from_parameters'
      ):
        continue
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      if attr.name == 'Phibdot':
        kwargs[attr.name] = 0.0
        continue
      kwargs[attr.name] = getattr(self, attr.name).get_value(t)
    return geometry_class(**kwargs)  # pytype: disable=wrong-keyword-args

  @functools.partial(jax_utils.jit, static_argnums=0)
  def __call__(self, t: chex.Numeric) -> Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, Geometry)


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometry(Geometry):
  """Circular geometry type used for testing only.

  Most users should default to using the Geometry class.
  """

  kappa: chex.Array
  kappa_face: chex.Array
  kappa_hires: chex.Array


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometryProvider(GeometryProvider):
  """Circular geometry type used for testing only.

  Most users should default to using the GeometryProvider class.
  """

  kappa: interpolated_param.InterpolatedVarSingleAxis
  kappa_face: interpolated_param.InterpolatedVarSingleAxis
  kappa_hires: interpolated_param.InterpolatedVarSingleAxis

  def __call__(self, t: chex.Numeric) -> Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, CircularAnalyticalGeometry)


@chex.dataclass(frozen=True)
class StandardGeometry(Geometry):
  """Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.
  """

  Ip_from_parameters: bool
  Ip: chex.Scalar
  psi: chex.Array
  psi_from_Ip: chex.Array
  jtot: chex.Array
  jtot_face: chex.Array
  delta_upper_face: chex.Array
  delta_lower_face: chex.Array


@chex.dataclass(frozen=True)
class StandardGeometryProvider(GeometryProvider):
  """Values to be interpolated for a Standard Geometry."""

  Ip_from_parameters: bool
  Ip: interpolated_param.InterpolatedVarSingleAxis
  psi: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip: interpolated_param.InterpolatedVarSingleAxis
  jtot: interpolated_param.InterpolatedVarSingleAxis
  jtot_face: interpolated_param.InterpolatedVarSingleAxis
  delta_upper_face: interpolated_param.InterpolatedVarSingleAxis
  delta_lower_face: interpolated_param.InterpolatedVarSingleAxis

  @functools.partial(jax_utils.jit, static_argnums=0)
  def __call__(self, t: chex.Numeric) -> Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, StandardGeometry)


def build_circular_geometry(
    nr: int = 25,
    kappa: float = 1.72,
    Rmaj: float = 6.2,
    Rmin: float = 2.0,
    B0: float = 5.3,
    hires_fac: int = 4,
) -> CircularAnalyticalGeometry:
  """Constructs a CircularAnalyticalGeometry.

  This is the standard entrypoint for building a circular geometry, not
  CircularAnalyticalGeometry.__init__(). chex.dataclasses do not allow
  overriding __init__ functions with different parameters than the attributes of
  the dataclass, so this builder function lives outside the class.

  Args:
    nr: Radial grid points (num cells)
    kappa: Elogination. Defaults to 1.72 for the ITER elongation, to
      approximately correct volume and area integral Jacobians.
    Rmaj: major radius (R) in meters
    Rmin: minor radius (a) in meters
    B0: Toroidal magnetic field on axis [T]
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A CircularAnalyticalGeometry instance.
  """
  # assumes that r/Rmin = rho_norm where rho is the toroidal flux coordinate
  # r_norm coordinate is r/Rmin in circular, and rho_norm in standard
  # geometry (CHEASE/EQDSK)
  # Define mesh (Slab Uniform 1D with Jacobian = 1)
  dr_norm = 1.0 / nr
  mesh = Grid1D.construct(nx=nr, dx=dr_norm)
  rmax = np.asarray(Rmin)
  # helper variables for mesh cells and faces
  # r coordinate of faces
  r_face_norm = mesh.face_centers
  # r coordinate of cell centers
  r_norm = mesh.cell_centers

  r_face = r_face_norm * rmax
  r = r_norm * rmax
  Rmaj = np.array(Rmaj)
  B0 = np.array(B0)

  # assumed elongation profile on cell grid
  kappa_param = kappa
  kappa = 1 + r_norm * (kappa_param - 1)
  # assumed elongation profile on cell grid
  kappa_face = 1 + r_face_norm * (kappa_param - 1)

  volume = 2 * np.pi**2 * Rmaj * r**2 * kappa
  volume_face = 2 * np.pi**2 * Rmaj * r_face**2 * kappa_face
  area = np.pi * r**2 * kappa
  area_face = np.pi * r_face**2 * kappa_face

  # V' = dV/drnorm for volume integrations
  vpr = 4 * np.pi**2 * Rmaj * r * kappa * rmax + volume / kappa * (
      kappa_param - 1
  )
  vpr_face = (
      4 * np.pi**2 * Rmaj * r_face * kappa_face * rmax
      + volume_face / kappa_face * (kappa_param - 1)
  )
  # pylint: disable=invalid-name
  # S' = dS/drnorm for area integrals on cell grid
  spr_cell = 2 * np.pi * r * kappa * rmax + area / kappa * (kappa_param - 1)
  spr_face = 2 * np.pi * r_face * kappa_face * rmax + area_face / kappa_face * (
      kappa_param - 1
  )

  delta_face = np.zeros(len(r_face))

  # Geometry variables for general geometry form of transport equations.
  # With circular geometry approximation.

  # g0: <\nabla V>
  g0 = vpr / rmax
  g0_face = vpr_face / rmax

  # g1: <(\nabla V)^2>
  g1 = vpr**2 / rmax**2
  g1_face = vpr_face**2 / rmax**2

  # g2: <(\nabla V)^2 / R^2>
  # V = 2*pi^2*R*r^2*kappa
  # Expand with small kappa expansion

  g2 = (
      16
      * np.pi**4
      * r**2
      * kappa**2
      * (1 + 0.5 * r_norm * (kappa_param - 1) / kappa_param) ** 2
  )
  g2_face = (
      16
      * np.pi**4
      * r_face**2
      * kappa_face**2
      * (1 + 0.5 * r_face_norm * (kappa_param - 1) / kappa_param) ** 2
  )

  # g3: <1/R^2> (done without a kappa correction)
  # <1/R^2> =
  # 1/2pi*int_0^2pi (1/(Rmaj+r*cosx)^2)dx =
  # 1/( Rmaj^2 * (1 - (r/Rmaj)^2)^3/2 )
  g3 = 1 / (Rmaj**2 * (1 - (r / Rmaj) ** 2) ** (3.0 / 2.0))
  g3_face = 1 / (Rmaj**2 * (1 - (r_face / Rmaj) ** 2) ** (3.0 / 2.0))

  # simplifying assumption for now, for J=R*B/(R0*B0)
  J = np.ones(len(r))
  J_face = np.ones(len(r_face))
  # simplified (constant) version of the F=B*R function
  F = np.ones(len(r)) * Rmaj * B0
  F_face = np.ones(len(r_face)) * Rmaj * B0

  # Using an approximation where:
  # g2g3_over_rhon = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc kappa assumption, which leads to more
  # reasonable values for g2g3_over_rhon through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rhon will be used.

  g2g3_over_rhon = 4 * np.pi**2 * vpr * g3 / (J * Rmaj)
  g2g3_over_rhon_face = 4 * np.pi**2 * vpr_face * g3_face / (J_face * Rmaj)

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current, which is
  # the only option for ad-hoc circular geometry.
  r_hires_norm = np.linspace(0, 1, nr * hires_fac)
  r_hires = r_hires_norm * rmax

  Rout = Rmaj + r
  Rout_face = Rmaj + r_face

  Rin = Rmaj - r
  Rin_face = Rmaj - r_face

  # assumed elongation profile on hires grid
  kappa_hires = 1 + r_hires_norm * (kappa_param - 1)

  volume_hires = 2 * np.pi**2 * Rmaj * r_hires**2 * kappa_hires
  area_hires = np.pi * r_hires**2 * kappa_hires

  # V' = dV/drnorm for volume integrations on hires grid
  vpr_hires = (
      4 * np.pi**2 * Rmaj * r_hires * kappa_hires * rmax
      + volume_hires / kappa_hires * (kappa_param - 1)
  )
  # S' = dS/drnorm for area integrals on hires grid
  spr_hires = (
      2 * np.pi * r_hires * kappa_hires * rmax
      + area_hires / kappa_hires * (kappa_param - 1)
  )

  g3_hires = 1 / (Rmaj**2 * (1 - (r_hires / Rmaj) ** 2) ** (3.0 / 2.0))
  F_hires = np.ones(len(r_hires)) * B0 * Rmaj
  g2g3_over_rhon_hires = 4 * np.pi**2 * vpr_hires * g3_hires * B0 / F_hires

  return CircularAnalyticalGeometry(
      # Set the standard geometry params.
      geometry_type=GeometryType.CIRCULAR.value,
      dr_norm=np.asarray(dr_norm),
      torax_mesh=mesh,
      rmax=rmax,
      Rmaj=Rmaj,
      Rmin=rmax,
      B0=B0,
      volume=volume,
      volume_face=volume_face,
      area=area,
      area_face=area_face,
      vpr=vpr,
      vpr_face=vpr_face,
      spr_cell=spr_cell,
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
      kappa=kappa,
      kappa_face=kappa_face,
      volume_hires=volume_hires,
      area_hires=area_hires,
      spr_hires=spr_hires,
      r_hires_norm=r_hires_norm,
      r_hires=r_hires,
      kappa_hires=kappa_hires,
      vpr_hires=vpr_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
  )


# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class StandardGeometryIntermediates:
  """Holds the intermediate values used to build a StandardGeometry.

  In particular these are the values that are used when interpolating different
  geometries.

  TODO(b/323504363): Specify the expected COCOS format.
  NOTE: Right now, TORAX does not have a specified COCOS format. Our team is
  working on adding this and updating documentation to make that clear. Of
  course, the CHEASE input data is COCOS 2, still.

  All inputs are 1D profiles vs normalized rho toroidal (rhon).

  Ip_from_paramaters: If True, the Ip is taken from the parameters and the
    values in the Geometry are resacled to match the new Ip.
  Rmaj: major radius (R) in meters. CHEASE geometries are normalized, so this
    is used as an unnormalization factor.
  Rmin: minor radius (a) in meters
  B: Toroidal magnetic field on axis [T].
  psi: Poloidal flux profile
  Ip_profile: Plasma current profile
  rho: Toroidal flux coordinate
  rhon: Normalized toroidal flux coordinate
  Rin: Radius of the flux surface at the inboard side at midplane
  Rout: Radius of the flux surface at the outboard side at midplane
  RBphi: Toroidal field flux function
  int_dl_over_Bp: oint (dl / Bp) (contour integral)
  flux_surf_avg_1_over_R2: <1/R**2>
  flux_surf_avg_Bp2: <Bp**2>
  flux_surf_avg_RBp: <R Bp>
  flux_surf_avg_R2Bp2: <R**2 Bp**2>
  delta_upper_face: Triangularity on upper face
  delta_lower_face: Triangularity on lower face
  volume: Volume profile
  area: Area profile
  nr: Radial grid points (num cells)
  hires_fac: Grid refinement factor for poloidal flux <--> plasma current
    calculations.
  """

  Ip_from_parameters: bool
  Rmaj: chex.Numeric
  Rmin: chex.Numeric
  B: chex.Numeric
  psi: chex.Array
  Ip_profile: chex.Array
  rho: chex.Array
  rhon: np.ndarray
  Rin: chex.Array
  Rout: chex.Array
  RBphi: chex.Array
  int_dl_over_Bp: chex.Array
  flux_surf_avg_1_over_R2: chex.Array
  flux_surf_avg_Bp2: chex.Array
  flux_surf_avg_RBp: chex.Array
  flux_surf_avg_R2Bp2: chex.Array
  delta_upper_face: chex.Array
  delta_lower_face: chex.Array
  volume: chex.Array
  area: chex.Array
  nr: int
  hires_fac: int

  @classmethod
  def from_chease(
      cls,
      geometry_dir: str | None = None,
      geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      Ip_from_parameters: bool = True,
      nr: int = 25,
      Rmaj: float = 6.2,
      Rmin: float = 2.0,
      B0: float = 5.3,
      hires_fac: int = 4,
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
        values in the Geometry are resacled to match the new Ip.
      nr: Radial grid points (num cells)
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
    chease_data = geometry_loader.load_chease_data(geometry_dir, geometry_file)

    # Prepare variables from CHEASE to be interpolated into our simulation
    # grid. CHEASE variables are normalized. Need to unnormalize them with
    # reference values poloidal flux and CHEASE-internal-calculated plasma
    # current.
    psiunnormfactor = Rmaj**2 * B0

    # set psi in TORAX units with 2*pi factor
    psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor * 2 * np.pi
    Ip_chease = chease_data['Ipprofile'] / constants.CONSTANTS.mu0 * Rmaj * B0

    # toroidal flux coordinate
    rho = chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] * Rmaj
    rhon = chease_data['RHO_TOR_NORM']
    # midplane radii
    Rin_chease = chease_data['R_INBOARD'] * Rmaj
    Rout_chease = chease_data['R_OUTBOARD'] * Rmaj
    # toroidal field flux function
    RBphi = chease_data['T=RBphi'] * Rmaj * B0

    int_dl_over_Bp = chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] * Rmaj / B0
    flux_surf_avg_1_over_R2 = chease_data['<1/R**2>'] / Rmaj**2
    flux_surf_avg_Bp2 = chease_data['<Bp**2>'] * B0**2
    flux_surf_avg_RBp = chease_data['<|grad(psi)|>'] * psiunnormfactor / Rmaj
    flux_surf_avg_R2Bp2 = (
        chease_data['<|grad(psi)|**2>'] * psiunnormfactor**2 / Rmaj**2
    )

    # volume, area, and dV/drho, dS/drho
    volume = chease_data['VOLUMEprofile'] * Rmaj**3
    area = chease_data['areaprofile'] * Rmaj**2

    return cls(
        Ip_from_parameters=Ip_from_parameters,
        Rmaj=Rmaj,
        Rmin=Rmin,
        B=B0,
        psi=psi,
        Ip_profile=Ip_chease,
        rho=rho,
        rhon=rhon,
        Rin=Rin_chease,
        Rout=Rout_chease,
        RBphi=RBphi,
        int_dl_over_Bp=int_dl_over_Bp,
        flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
        flux_surf_avg_Bp2=flux_surf_avg_Bp2,
        flux_surf_avg_RBp=flux_surf_avg_RBp,
        flux_surf_avg_R2Bp2=flux_surf_avg_R2Bp2,
        delta_upper_face=chease_data['delta_upper'],
        delta_lower_face=chease_data['delta_bottom'],
        volume=volume,
        area=area,
        nr=nr,
        hires_fac=hires_fac,
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
  g2g3_over_rhon = g2[1:] * g3[1:] / intermediate.rhon[1:]
  g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))

  J = intermediate.RBphi / (intermediate.Rmaj * intermediate.B)

  # make an alternative initial psi, self-consistent with CHEASE Ip profile
  # needed because CHEASE psi profile has noisy second derivatives
  dpsidrho = (
      intermediate.Ip_profile[1:]
      * (16 * constants.CONSTANTS.mu0 * np.pi**4 * intermediate.rho[-1])
      / (g2g3_over_rhon[1:] * intermediate.Rmaj * J[1:])
  )
  dpsidrho = np.concatenate((np.zeros(1), dpsidrho))
  psi_from_Ip = scipy.integrate.cumulative_trapezoid(
      y=dpsidrho, x=intermediate.rho, initial=0.0
  )

  # set Ip-consistent psi derivative boundary condition (although will be
  # replaced later with an fvm constraint)
  psi_from_Ip[-1] = psi_from_Ip[-2] + (
      16 * constants.CONSTANTS.mu0 * np.pi**4 * intermediate.rho[-1]
  ) * intermediate.Ip_profile[-1] / (
      g2g3_over_rhon[-1] * intermediate.Rmaj * J[-1]
  ) * (
      intermediate.rho[-1] - intermediate.rho[-2]
  )

  # dV/drhon, dS/drhon
  vpr = np.gradient(intermediate.volume, intermediate.rhon)
  spr = np.gradient(intermediate.area, intermediate.rhon)
  # gradient boundary approximation not appropriate here
  vpr[0] = 0
  spr[0] = 0

  # plasma current density
  jtot = (
      2
      * np.pi
      * intermediate.Rmaj
      * np.gradient(intermediate.Ip_profile, intermediate.volume)
  )

  # fill geometry structure
  # r_norm coordinate is rho_tor_norm

  # fill geometry structure
  # r_norm coordinate is rho_tor_norm
  dr_norm = float(intermediate.rhon[-1]) / intermediate.nr
  # normalized grid
  mesh = Grid1D.construct(nx=intermediate.nr, dx=dr_norm)
  rmax = intermediate.rho[-1]  # radius denormalization constant
  # helper variables for mesh cells and faces
  r_face_norm = mesh.face_centers
  r_norm = mesh.cell_centers

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current.
  r_hires_norm = np.linspace(0, 1, intermediate.nr * intermediate.hires_fac)
  r_hires = r_hires_norm * rmax

  rhon_interpolation_func = lambda x, y: np.interp(x, intermediate.rhon, y)
  # V' for volume integrations on face grid
  vpr_face = rhon_interpolation_func(r_face_norm, vpr)
  # V' for volume integrations on cell grid
  vpr_hires = rhon_interpolation_func(r_hires_norm, vpr)
  vpr = rhon_interpolation_func(r_norm, vpr)

  # S' for area integrals on face grid
  spr_face = rhon_interpolation_func(r_face_norm, spr)
  # S' for area integrals on cell grid
  spr_cell = rhon_interpolation_func(r_norm, spr)
  spr_hires = rhon_interpolation_func(r_hires_norm, spr)

  # triangularity on cell grid
  delta_upper_face = rhon_interpolation_func(
      r_face_norm, intermediate.delta_upper_face
  )
  delta_lower_face = rhon_interpolation_func(
      r_face_norm, intermediate.delta_lower_face
  )

  # average triangularity
  delta_face = 0.5 * (delta_upper_face + delta_lower_face)

  F_face = rhon_interpolation_func(r_face_norm, intermediate.RBphi)
  F = rhon_interpolation_func(r_norm, intermediate.RBphi)
  F_hires = rhon_interpolation_func(r_hires_norm, intermediate.RBphi)

  psi = rhon_interpolation_func(r_norm, intermediate.psi)
  psi_from_Ip = rhon_interpolation_func(r_norm, psi_from_Ip)

  jtot_face = rhon_interpolation_func(r_face_norm, jtot)
  jtot = rhon_interpolation_func(r_norm, jtot)

  Rin_face = rhon_interpolation_func(r_face_norm, intermediate.Rin)
  Rin = rhon_interpolation_func(r_norm, intermediate.Rin)

  Rout_face = rhon_interpolation_func(r_face_norm, intermediate.Rout)
  Rout = rhon_interpolation_func(r_norm, intermediate.Rout)

  g0_face = rhon_interpolation_func(r_face_norm, g0)
  g0 = rhon_interpolation_func(r_norm, g0)

  g1_face = rhon_interpolation_func(r_face_norm, g1)
  g1 = rhon_interpolation_func(r_norm, g1)

  g2_face = rhon_interpolation_func(r_face_norm, g2)
  g2 = rhon_interpolation_func(r_norm, g2)

  g3_face = rhon_interpolation_func(r_face_norm, g3)
  g3 = rhon_interpolation_func(r_norm, g3)

  g2g3_over_rhon_face = rhon_interpolation_func(r_face_norm, g2g3_over_rhon)
  g2g3_over_rhon_hires = rhon_interpolation_func(r_hires_norm, g2g3_over_rhon)
  g2g3_over_rhon = rhon_interpolation_func(r_norm, g2g3_over_rhon)

  volume_face = rhon_interpolation_func(r_face_norm, intermediate.volume)
  volume_hires = rhon_interpolation_func(r_hires_norm, intermediate.volume)
  volume = rhon_interpolation_func(r_norm, intermediate.volume)

  area_face = rhon_interpolation_func(r_face_norm, intermediate.area)
  area_hires = rhon_interpolation_func(r_hires_norm, intermediate.area)
  area = rhon_interpolation_func(r_norm, intermediate.area)

  return StandardGeometry(
      geometry_type=GeometryType.CHEASE.value,
      dr_norm=np.asarray(dr_norm),
      torax_mesh=mesh,
      rmax=rmax,
      Rmaj=intermediate.Rmaj,
      Rmin=intermediate.Rmin,
      B0=intermediate.B,
      volume=volume,
      volume_face=volume_face,
      area=area,
      area_face=area_face,
      vpr=vpr,
      vpr_face=vpr_face,
      spr_cell=spr_cell,
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
      Ip=intermediate.Ip_profile[-1],
      psi=psi,
      psi_from_Ip=psi_from_Ip,
      jtot=jtot,
      jtot_face=jtot_face,
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
      volume_hires=volume_hires,
      area_hires=area_hires,
      spr_hires=spr_hires,
      r_hires_norm=r_hires_norm,
      r_hires=r_hires,
      vpr_hires=vpr_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
  )


# pylint: enable=invalid-name
