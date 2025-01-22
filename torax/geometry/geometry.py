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
import contourpy
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from torax import array_typing
from torax import constants
from torax import interpolated_param
from torax import jax_utils
from torax.geometry import geometry_loader


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
  FBT = 2
  EQDSK = 3


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
  drho_norm: array_typing.ArrayFloat
  Phi: chex.Array
  Phi_face: chex.Array
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
  elongation: chex.Array
  elongation_face: chex.Array
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
  rho_hires_norm: chex.Array
  rho_hires: chex.Array
  vpr_hires: chex.Array
  Phibdot: chex.Array
  _z_magnetic_axis: chex.Array | None

  @property
  def rho_norm(self) -> chex.Array:
    return self.torax_mesh.cell_centers

  @property
  def rho_face_norm(self) -> chex.Array:
    return self.torax_mesh.face_centers

  @property
  def rho_face(self) -> chex.Array:
    return self.rho_face_norm * self.rho_b

  @property
  def rho(self) -> chex.Array:
    return self.rho_norm * self.rho_b

  @property
  def rmid(self) -> chex.Array:
    return (self.Rout - self.Rin) / 2

  @property
  def rmid_face(self) -> chex.Array:
    return (self.Rout_face - self.Rin_face) / 2

  @property
  def drho(self) -> chex.Array:
    return self.drho_norm * self.rho_b

  @property
  def rho_b(self) -> chex.Array:
    """Toroidal flux coordinate at boundary (LCFS)."""
    return jnp.sqrt(self.Phib / np.pi / self.B0)

  @property
  def Phib(self) -> chex.Array:
    """Toroidal flux at boundary (LCFS)."""
    return self.Phi_face[-1]

  @property
  def g1_over_vpr(self) -> chex.Array:
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1) / self.rho_b,  # correct value is 1/rho_b on-axis
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
        jnp.ones(1) / self.rho_b**2,  # correct value is 1/rho_b**2 on-axis
        self.g1_face[1:] / self.vpr_face[1:] ** 2,  # avoid div by zero on-axis
    ))

  @property
  def z_magnetic_axis(self) -> chex.Numeric:
    z_magnetic_axis = self._z_magnetic_axis
    if z_magnetic_axis is not None:
      return z_magnetic_axis
    else:
      raise RuntimeError(
          'Geometry does not have a z magnetic axis.'
      )


@chex.dataclass(frozen=True)
class GeometryProvider:
  """A geometry which holds variables to interpolated based on time."""

  geometry_type: int
  torax_mesh: Grid1D
  drho_norm: interpolated_param.InterpolatedVarSingleAxis
  Phi: interpolated_param.InterpolatedVarSingleAxis
  Phi_face: interpolated_param.InterpolatedVarSingleAxis
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
  elongation: interpolated_param.InterpolatedVarSingleAxis
  elongation_face: interpolated_param.InterpolatedVarSingleAxis
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
  rho_hires_norm: interpolated_param.InterpolatedVarSingleAxis
  rho_hires: interpolated_param.InterpolatedVarSingleAxis
  vpr_hires: interpolated_param.InterpolatedVarSingleAxis
  _z_magnetic_axis: interpolated_param.InterpolatedVarSingleAxis | None

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
      if attr.name == '_z_magnetic_axis':
        if initial_geometry._z_magnetic_axis is None:  # pylint: disable=protected-access
          kwargs[attr.name] = None
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
      if attr.name == '_z_magnetic_axis':
        if self._z_magnetic_axis is None:
          kwargs[attr.name] = None
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

  elongation_hires: chex.Array


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometryProvider(GeometryProvider):
  """Circular geometry type used for testing only.

  Most users should default to using the GeometryProvider class.
  """

  elongation_hires: interpolated_param.InterpolatedVarSingleAxis

  def __call__(self, t: chex.Numeric) -> Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, CircularAnalyticalGeometry)


@chex.dataclass(frozen=True)
class StandardGeometry(Geometry):
  """Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.
  """

  Ip_from_parameters: bool
  Ip_profile_face: chex.Array
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
  Ip_profile_face: interpolated_param.InterpolatedVarSingleAxis
  psi: interpolated_param.InterpolatedVarSingleAxis
  psi_from_Ip: interpolated_param.InterpolatedVarSingleAxis
  jtot: interpolated_param.InterpolatedVarSingleAxis
  jtot_face: interpolated_param.InterpolatedVarSingleAxis
  delta_upper_face: interpolated_param.InterpolatedVarSingleAxis
  delta_lower_face: interpolated_param.InterpolatedVarSingleAxis
  elongation: interpolated_param.InterpolatedVarSingleAxis
  elongation_face: interpolated_param.InterpolatedVarSingleAxis

  @functools.partial(jax_utils.jit, static_argnums=0)
  def __call__(self, t: chex.Numeric) -> Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, StandardGeometry)


def build_circular_geometry(
    n_rho: int = 25,
    elongation_LCFS: float = 1.72,
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
    n_rho: Radial grid points (num cells)
    elongation_LCFS: Elongation at last closed flux surface. Defaults to 1.72
      for the ITER elongation, to approximately correct volume and area integral
      Jacobians.
    Rmaj: major radius (R) in meters
    Rmin: minor radius (a) in meters
    B0: Toroidal magnetic field on axis [T]
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A CircularAnalyticalGeometry instance.
  """
  # circular geometry assumption of r/Rmin = rho_norm, the normalized
  # toroidal flux coordinate.
  drho_norm = 1.0 / n_rho
  # Define mesh (Slab Uniform 1D with Jacobian = 1)
  mesh = Grid1D.construct(nx=n_rho, dx=drho_norm)
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
  spr_cell = 2 * np.pi * rho * elongation * rho_b + area / elongation * (
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

  return CircularAnalyticalGeometry(
      # Set the standard geometry params.
      geometry_type=GeometryType.CIRCULAR.value,
      drho_norm=np.asarray(drho_norm),
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
      elongation=elongation,
      elongation_face=elongation_face,
      volume_hires=volume_hires,
      area_hires=area_hires,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      elongation_hires=elongation_hires,
      vpr_hires=vpr_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
      _z_magnetic_axis=np.asarray(0.0),
  )


# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class StandardGeometryIntermediates:
  """Holds the intermediate values used to build a StandardGeometry.

  In particular these are the values that are used when interpolating different
  geometries.

  TODO(b/335204606): Specify the expected COCOS format.
  NOTE: Right now, TORAX does not have a specified COCOS format. Our team is
  working on adding this and updating documentation to make that clear. Of
  course, the CHEASE input data is COCOS 2, still.

  All inputs are 1D profiles vs normalized rho toroidal (rhon).

  Ip_from_parameters: If True, the Ip is taken from the parameters and the
    values in the Geometry are resacled to match the new Ip.
  Rmaj: major radius (R) in meters. CHEASE geometries are normalized, so this
    is used as an unnormalization factor.
  Rmin: minor radius (a) in meters
  B: Toroidal magnetic field on axis [T].
  psi: Poloidal flux profile
  Ip_profile: Plasma current profile
  Phi: Toroidal flux profile
  Rin: Radius of the flux surface at the inboard side at midplane
  Rout: Radius of the flux surface at the outboard side at midplane
  F: Toroidal field flux function
  int_dl_over_Bp: 1/ oint (dl / Bp) (contour integral)
  flux_surf_avg_1_over_R2: <1/R**2>
  flux_surf_avg_Bp2: <Bp**2>
  flux_surf_avg_RBp: <R Bp>
  flux_surf_avg_R2Bp2: <R**2 Bp**2>
  delta_upper_face: Triangularity on upper face
  delta_lower_face: Triangularity on lower face
  elongation: Plasma elongation profile
  vpr: dVolume/drhonorm profile
  n_rho: Radial grid points (num cells)
  hires_fac: Grid refinement factor for poloidal flux <--> plasma current
    calculations.
  z_magnetic_axis: z position of magnetic axis [m]
  """

  geometry_type: GeometryType
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
    """Extrapolates edge values based on a Cubic spline fit."""
    # Check if last flux surface is diverted and correct if so
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

  @classmethod
  def from_chease(
      cls,
      geometry_dir: str | None = None,
      geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      Ip_from_parameters: bool = True,
      n_rho: int = 25,
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
        geometry_type=GeometryType.CHEASE,
        Ip_from_parameters=Ip_from_parameters,
        Rmaj=Rmaj,
        Rmin=Rmin,
        B=B0,
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
        geometry_type=GeometryType.FBT,
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
      geometry_dir: str | None = None,
      geometry_file: str = 'EQDSK_ITERhybrid_COCOS02.eqdsk',
      hires_fac: int = 4,
      Ip_from_parameters: bool = True,
      n_rho: int = 25,
      n_surfaces: int = 100,
      last_surface_factor: float = 0.99,
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
        geometry_type=GeometryType.EQDSK,
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
  mesh = Grid1D.construct(nx=intermediate.n_rho, dx=drho_norm)
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
  vpr_hires = rhon_interpolation_func(rho_hires_norm, vpr)
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
  elongation = rhon_interpolation_func(
      rho_norm, intermediate.elongation
  )
  elongation_face = rhon_interpolation_func(
      rho_face_norm, intermediate.elongation
  )

  Phi_face = rhon_interpolation_func(rho_face_norm, intermediate.Phi)
  Phi = rhon_interpolation_func(rho_norm, intermediate.Phi)

  F_face = rhon_interpolation_func(rho_face_norm, intermediate.F)
  F = rhon_interpolation_func(rho_norm, intermediate.F)
  F_hires = rhon_interpolation_func(rho_hires_norm, intermediate.F)

  psi = rhon_interpolation_func(rho_norm, intermediate.psi)
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
  volume_hires = rhon_interpolation_func(rho_hires_norm, volume_intermediate)
  volume = rhon_interpolation_func(rho_norm, volume_intermediate)

  area_face = rhon_interpolation_func(rho_face_norm, area_intermediate)
  area_hires = rhon_interpolation_func(rho_hires_norm, area_intermediate)
  area = rhon_interpolation_func(rho_norm, area_intermediate)

  return StandardGeometry(
      geometry_type=intermediate.geometry_type.value,
      drho_norm=np.asarray(drho_norm),
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
      Ip_profile_face=Ip_profile_face,
      psi=psi,
      psi_from_Ip=psi_from_Ip,
      jtot=jtot,
      jtot_face=jtot_face,
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
      elongation=elongation,
      elongation_face=elongation_face,
      volume_hires=volume_hires,
      area_hires=area_hires,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      vpr_hires=vpr_hires,
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
    raise ValueError("L data is missing the 't' key.")

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


# pylint: enable=invalid-name
