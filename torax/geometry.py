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

import dataclasses
import enum

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from torax import constants
from torax import geometry_loader
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
  dx: chex.Array
  face_centers: chex.Array
  cell_centers: chex.Array

  def __post_init__(self):
    jax_utils.assert_rank(self.nx, 0)
    jax_utils.assert_rank(self.dx, 0)
    jax_utils.assert_rank(self.face_centers, 1)
    jax_utils.assert_rank(self.cell_centers, 1)

  @classmethod
  def construct(cls, nx: int, dx: chex.Array) -> Grid1D:
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

  geometry_type: int
  dr_norm: chex.Array
  mesh: Grid1D
  rmax: chex.Array
  r_face_norm: chex.Array
  r_norm: chex.Array
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
  g2g3_over_rho: chex.Array
  g2g3_over_rho_face: chex.Array
  g2g3_over_rho_hires: chex.Array
  J: chex.Array
  J_face: chex.Array
  J_hires: chex.Array
  F: chex.Array
  F_face: chex.Array
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

  @property
  def r_face(self) -> chex.Array:
    return self.r_face_norm * self.rmax

  @property
  def r(self) -> chex.Array:
    return self.r_norm * self.rmax

  @property
  def dr(self) -> chex.Array:
    return self.dr_norm * self.rmax

  @property
  def g1_over_vpr(self) -> chex.Array:
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1),  # correct value is unity on-axis
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
        jnp.ones(1),  # correct value is unity on-axis
        self.g1_face[1:] / self.vpr_face[1:] ** 2,  # avoid div by zero on-axis
    ))


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometry(Geometry):
  """Circular geometry type used for testing only.

  Most users should default to using the Geometry class.
  """

  kappa: chex.Array
  kappa_face: chex.Array
  kappa_hires: chex.Array


@chex.dataclass(frozen=True)
class StandardGeometry(Geometry):
  """Standard geometry object including additional useful attributes, like psi.

  Most instances of Geometry should be of this type.
  """

  g2: chex.Array
  g3: chex.Array
  psi: chex.Array
  psi_from_Ip: chex.Array
  jtot: chex.Array
  jtot_face: chex.Array
  delta_upper_face: chex.Array
  delta_lower_face: chex.Array


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
  dr_norm = np.array(1) / nr
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

  # V' for volume integrations
  vpr = (
      4 * np.pi**2 * Rmaj * r * kappa
      + volume / kappa * (kappa_param - 1) / rmax
  )
  vpr_face = (
      4 * np.pi**2 * Rmaj * r_face * kappa_face
      + volume_face / kappa_face * (kappa_param - 1) / rmax
  )
  # pylint: disable=invalid-name
  # S' for area integrals on cell grid
  spr_cell = 2 * np.pi * r * kappa + area / kappa * (kappa_param - 1) / rmax
  spr_face = (
      2 * np.pi * r_face * kappa_face
      + area_face / kappa_face * (kappa_param - 1) / rmax
  )

  delta_face = np.zeros(len(r_face))

  # Geometry variables for general geometry form of transport equations.
  # With circular geometry approximation.

  # g0: <\nabla V>
  g0 = vpr
  g0_face = vpr_face

  # g1: <\nabla V^2>
  g1 = vpr**2
  g1_face = vpr_face**2

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
  # g2g3_over_rho = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc kappa assumption, which leads to more
  # reasonable values for g2g3_over_rho through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rho will be used.

  g2g3_over_rho = 4 * np.pi**2 * vpr * g3 / (J * Rmaj)
  g2g3_over_rho_face = 4 * np.pi**2 * vpr_face * g3_face / (J_face * Rmaj)

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

  # V' for volume integrations on hires grid
  vpr_hires = (
      4 * np.pi**2 * Rmaj * r_hires * kappa_hires
      + volume_hires / kappa_hires * (kappa_param - 1) / rmax
  )
  # S' for area integrals on hires grid
  spr_hires = (
      2 * np.pi * r_hires * kappa_hires
      + area_hires / kappa_hires * (kappa_param - 1) / rmax
  )

  g3_hires = 1 / (Rmaj**2 * (1 - (r_hires / Rmaj) ** 2) ** (3.0 / 2.0))
  J_hires = np.ones(len(r_hires))
  g2g3_over_rho_hires = 4 * np.pi**2 * vpr_hires * g3_hires / (J_hires * Rmaj)

  return CircularAnalyticalGeometry(
      # Set the standard geometry params.
      geometry_type=GeometryType.CIRCULAR.value,
      dr_norm=dr_norm,
      mesh=mesh,
      rmax=rmax,
      r_face_norm=r_face_norm,
      r_norm=r_norm,
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
      g2g3_over_rho=g2g3_over_rho,
      g2g3_over_rho_face=g2g3_over_rho_face,
      g2g3_over_rho_hires=g2g3_over_rho_hires,
      J=J,
      J_face=J_face,
      J_hires=J_hires,
      F=F,
      F_face=F_face,
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

  Rmaj: major radius (R) in meters. CHEASE geometries are normalized, so this
    is used as an unnormalization factor.
  Rmin: minor radius (a) in meters
  B: Toroidal magnetic field on axis [T].
  psi: Poloidal flux profile
  Ip: Plasma current profile
  rho: Midplane radii
  rhon: Toroidal flux coordinate
  Rin: Midplane radii
  Rout: Midplane radii
  RBphi: Toroidal field flux function
  int_Jdchi: <|grad(psi)|>
  flux_norm_1_over_R2: <1/R**2>
  flux_norm_Bp2: <Bp**2>
  flux_norm_dpsi: <|grad(psi)|>
  flux_norm_dpsi2: <|grad(psi)|**2>
  delta_upper_face: Triangularity on upper face
  delta_lower_face: Triangularity on lower face
  volume: Volume profile
  area: Area profile
  nr: Radial grid points (num cells)
  hires_fac: Grid refinement factor for poloidal flux <--> plasma current
    calculations.
  """
  Rmaj: chex.Numeric
  Rmin: chex.Numeric
  B: chex.Numeric
  psi: chex.Array
  Ip: chex.Array
  rho: chex.Array
  rhon: chex.Array
  Rin: chex.Array
  Rout: chex.Array
  RBphi: chex.Array
  int_Jdchi: chex.Array
  flux_norm_1_over_R2: chex.Array
  flux_norm_Bp2: chex.Array
  flux_norm_dpsi: chex.Array
  flux_norm_dpsi2: chex.Array
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
    psiunnormfactor = (Rmaj**2 * B0) * 2 * np.pi
    psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor
    Ip_chease = chease_data['Ipprofile'] / constants.CONSTANTS.mu0 * Rmaj * B0

    # toroidal flux coordinate
    rho = chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] * Rmaj
    rhon = chease_data['RHO_TOR_NORM']
    # midplane radii
    Rin_chease = chease_data['R_INBOARD'] * Rmaj
    Rout_chease = chease_data['R_OUTBOARD'] * Rmaj
    # toroidal field flux function
    RBphi = chease_data['T=RBphi'] * Rmaj * B0

    int_Jdchi = chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] * Rmaj / B0
    flux_norm_1_over_R2 = chease_data['<1/R**2>'] / Rmaj**2
    flux_norm_Bp2 = chease_data['<Bp**2>'] * B0**2 * 4 * np.pi**2
    flux_norm_dpsi = chease_data['<|grad(psi)|>'] * Rmaj * B0 * 2 * np.pi
    flux_norm_dpsi2 = (
        chease_data['<|grad(psi)|**2>'] * (Rmaj * B0) ** 2 * 4 * np.pi**2
    )

    # volume, area, and dV/drho, dS/drho
    volume = chease_data['VOLUMEprofile'] * Rmaj**3
    area = chease_data['areaprofile'] * Rmaj**2

    return cls(
        Rmaj=Rmaj,
        Rmin=Rmin,
        B=B0,
        psi=psi,
        Ip=Ip_chease,
        rho=rho,
        rhon=rhon,
        Rin=Rin_chease,
        Rout=Rout_chease,
        RBphi=RBphi,
        int_Jdchi=int_Jdchi,
        flux_norm_1_over_R2=flux_norm_1_over_R2,
        flux_norm_Bp2=flux_norm_Bp2,
        flux_norm_dpsi=flux_norm_dpsi,
        flux_norm_dpsi2=flux_norm_dpsi2,
        delta_upper_face=chease_data['delta_upper'],
        delta_lower_face=chease_data['delta_bottom'],
        volume=volume,
        area=area,
        nr=nr,
        hires_fac=hires_fac,
    )


def build_standard_geometry(
    intermediate: StandardGeometryIntermediates
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
  C1 = intermediate.int_Jdchi
  C2 = intermediate.flux_norm_1_over_R2 * C1
  C3 = intermediate.flux_norm_Bp2 * C1
  C4 = intermediate.flux_norm_dpsi2 * C1

  # derived quantities for transport equations and transformations

  g0 = intermediate.flux_norm_dpsi * C1  # <\nabla V>
  g1 = C1 * C4  # <(\nabla V)**2>
  g2 = C1 * C3  # <(\nabla V)**2 / R**2>
  g3 = C2[1:] / C1[1:]  # <1/R**2>
  g3 = np.concatenate((np.array([1 / intermediate.Rin[0] ** 2]), g3))
  g2g3_over_rho = g2[1:] * g3[1:] / intermediate.rho[1:]
  g2g3_over_rho = np.concatenate((np.zeros(1), g2g3_over_rho))

  J = intermediate.RBphi / (intermediate.Rmaj * intermediate.B)

  # make an alternative initial psi, self-consistent with CHEASE Ip profile
  # needed because CHEASE psi profile has noisy second derivatives
  dpsidrho = (
      intermediate.Ip[1:]
      * (16 * constants.CONSTANTS.mu0 * np.pi**4)
      / (g2g3_over_rho[1:] * intermediate.Rmaj * J[1:])
  )
  dpsidrho = np.concatenate((np.zeros(1), dpsidrho))
  psi_from_Ip = np.zeros(len(intermediate.psi))
  for i in range(1, len(psi_from_Ip) + 1):
    psi_from_Ip[i - 1] = scipy.integrate.trapezoid(
        dpsidrho[:i], intermediate.rho[:i]
    )
  # set Ip-consistent psi derivative boundary condition (although will be
  # replaced later with an fvm constraint)
  psi_from_Ip[-1] = psi_from_Ip[-2] + (
      16 * constants.CONSTANTS.mu0 * np.pi**4
  ) * intermediate.Ip[-1] / (g2g3_over_rho[-1] * intermediate.Rmaj * J[-1]) * (
      intermediate.rho[-1] - intermediate.rho[-2]
  )

  # dV/drho, dS/drho
  vpr = np.gradient(intermediate.volume, intermediate.rho)
  spr = np.gradient(intermediate.area, intermediate.rho)
  # gradient boundary approximation not appropriate here
  vpr[0] = 0
  spr[0] = 0

  # plasma current density
  jtot = (
      2
      * np.pi
      * intermediate.Rmaj
      * np.gradient(intermediate.Ip, intermediate.volume)
  )

  # fill geometry structure
  # r_norm coordinate is rho_tor_norm

  # fill geometry structure
  # r_norm coordinate is rho_tor_norm
  dr_norm = intermediate.rhon[-1] / intermediate.nr
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

  interp_func = lambda x: np.interp(x, intermediate.rhon, vpr)
  # V' for volume integrations on face grid
  vpr_face = interp_func(r_face_norm)
  # V' for volume integrations on cell grid
  vpr_hires = interp_func(r_hires_norm)
  vpr = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, spr)
  # S' for area integrals on face grid
  spr_face = interp_func(r_face_norm)
  # S' for area integrals on cell grid
  spr_cell = interp_func(r_norm)
  spr_hires = interp_func(r_hires_norm)

  # triangularity on cell grid
  interp_func = lambda x: np.interp(x, intermediate.rhon,
                                    intermediate.delta_upper_face)
  delta_upper_face = interp_func(r_face_norm)
  interp_func = lambda x: np.interp(x, intermediate.rhon,
                                    intermediate.delta_lower_face)
  delta_lower_face = interp_func(r_face_norm)

  # average triangularity
  delta_face = 0.5 * (delta_upper_face + delta_lower_face)

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.RBphi)
  F_face = interp_func(r_face_norm)
  F_hires = interp_func(r_hires_norm)
  F = interp_func(r_norm)
  # Normalized toroidal flux function
  J = F / intermediate.Rmaj / intermediate.B
  J_face = F_face / intermediate.Rmaj / intermediate.B
  J_hires = F_hires / intermediate.Rmaj / intermediate.B

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.psi)
  psi = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, psi_from_Ip)
  psi_from_Ip = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, jtot)
  jtot_face = interp_func(r_face_norm)
  jtot = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.Rin)
  Rin_face = interp_func(r_face_norm)
  Rin = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.Rout)
  Rout_face = interp_func(r_face_norm)
  Rout = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, g0)
  g0_face = interp_func(r_face_norm)
  g0 = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, g1)
  g1_face = interp_func(r_face_norm)
  g1 = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, g2)
  g2_face = interp_func(r_face_norm)
  g2 = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, g3)
  g3_face = interp_func(r_face_norm)
  g3 = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, g2g3_over_rho)
  g2g3_over_rho_face = interp_func(r_face_norm)
  g2g3_over_rho_hires = interp_func(r_hires_norm)
  g2g3_over_rho = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.volume)
  volume_face = interp_func(r_face_norm)
  volume_hires = interp_func(r_hires_norm)
  volume = interp_func(r_norm)

  interp_func = lambda x: np.interp(x, intermediate.rhon, intermediate.area)
  area_face = interp_func(r_face_norm)
  area_hires = interp_func(r_hires_norm)
  area = interp_func(r_norm)

  return StandardGeometry(
      geometry_type=GeometryType.CHEASE.value,
      dr_norm=dr_norm,
      mesh=mesh,
      rmax=rmax,
      r_face_norm=r_face_norm,
      r_norm=r_norm,
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
      g2g3_over_rho=g2g3_over_rho,
      g2g3_over_rho_face=g2g3_over_rho_face,
      g2g3_over_rho_hires=g2g3_over_rho_hires,
      J=J,
      J_face=J_face,
      J_hires=J_hires,
      F=F,
      F_face=F_face,
      Rin=Rin,
      Rin_face=Rin_face,
      Rout=Rout,
      Rout_face=Rout_face,
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
  )


# pylint: enable=invalid-name
