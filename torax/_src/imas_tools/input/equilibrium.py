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

"""Input mapping functions for use of IMAS equilibrium IDSs with TORAX."""
from collections.abc import Mapping
import logging
from typing import Any

from imas import ids_toplevel
import numpy as np
import scipy
from torax._src.imas_tools.input import loader


# TODO(b/379832500) - Modify for consistency when we have a fixed TORAX COCOS.
# pylint: disable=invalid-name


def _load_equilibrium(
    geometry_directory: str | None = None,
    equilibrium_object: ids_toplevel.IDSToplevel | None = None,
    imas_uri: str | None = None,
    imas_filepath: str | None = None,
    explicit_convert: bool = False,
) -> ids_toplevel.IDSToplevel:
  """Loads an equilibrium IDS from the given source.

  Args:
    geometry_directory: Directory where to find the equilibrium object. If None,
      it defaults to another dir. See `load_geo_data` implementation.
    equilibrium_object: The equilibrium IDS containing the relevant data.
    imas_uri: The IMAS uri containing the equilibrium data.
    imas_filepath: The path to the IMAS netCDF file containing the equilibrium
      data.
    explicit_convert: Whether to explicitly convert the IDS to the current DD
      version. If True, an explicit conversion will be attempted. Explicit
      conversion is recommended when converting between major DD versions.
      https://imas-python.readthedocs.io/en/latest/multi-dd.html#conversion-of-idss-between-dd-versions

  Returns:
    The loaded equilibrium IDS.

  Raises:
    ValueError: If none of the three input sources are provided.
    TypeError: If the loaded IDS is not an equilibrium IDS.
  """
  if equilibrium_object is not None:
    equilibrium = equilibrium_object
  elif imas_uri is not None:
    equilibrium = loader.load_imas_data(
        imas_uri, "equilibrium", geometry_directory, explicit_convert
    )
  elif imas_filepath is not None:
    equilibrium = loader.load_imas_data(
        imas_filepath, "equilibrium", geometry_directory, explicit_convert
    )
  else:
    raise ValueError(
        "equilibrium_object must be a string (file path) or an IDS"
    )

  if equilibrium.metadata.name != "equilibrium":
    raise TypeError(
        f"Expected equilibrium IDS, got {equilibrium.metadata.name} IDS."
    )
  return equilibrium


def _geometry_from_single_slice(
    equilibrium: ids_toplevel.IDSToplevel,
    face_centers: np.ndarray,
    Ip_from_parameters: bool = False,
    hires_factor: int = 4,
    slice_index: int = 0,
) -> dict[str, Any]:
  """Extracts geometry data from a single time slice of an equilibrium IDS.

  Args:
    equilibrium: The equilibrium IDS.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    slice_index: Index of the time slice to process.

  Returns:
    A dict of intermediate geometry values for building a StandardGeometry.
  """
  IMAS_data = equilibrium.time_slice[slice_index]
  # IMAS python API returns custom primitive types (e.g. IDSFloat0D,
  # IDSNumericArray) instead of standard python floats or numpy arrays. We must
  # cast these to standard numpy arrays using np.asarray() (or via implicit
  # numpy operations like np.abs), otherwise JAX JIT compilation will fail with
  # TypeErrors during PyTree tracing.

  # Poloidal flux.
  psi = np.asarray(IMAS_data.profiles_1d.psi)

  # Toroidal flux.
  phi = np.asarray(IMAS_data.profiles_1d.phi)

  # Midplane radii.
  R_in = IMAS_data.profiles_1d.r_inboard
  R_out = IMAS_data.profiles_1d.r_outboard
  R_major_profile = (R_in + R_out) / 2.0

  # R_major is the geometric center of the LCFS.
  R_major = np.asarray(R_major_profile[-1])

  # IMAS defines the vacuum toroidal field b0 at the reference radius r0.
  # Scale to R_major using B_vac ∝ 1/R, consistent with EQDSK/FBT loaders.
  r0 = np.asarray(equilibrium.vacuum_toroidal_field.r0)
  B_0 = np.abs(equilibrium.vacuum_toroidal_field.b0[0]) * r0 / R_major

  # toroidal field flux function
  F = np.asarray(IMAS_data.profiles_1d.f)

  # Flux surface integrals of various geometry quantities.
  # IDS Contour integrals.
  if IMAS_data.profiles_1d.dvolume_dpsi:
    dvoldpsi = np.asarray(IMAS_data.profiles_1d.dvolume_dpsi)
  else:
    dvoldpsi = np.gradient(
        IMAS_data.profiles_1d.volume, IMAS_data.profiles_1d.psi
    )
  # dpsi_drho_tor
  if IMAS_data.profiles_1d.dpsi_drho_tor:
    dpsidrhotor = np.asarray(IMAS_data.profiles_1d.dpsi_drho_tor)
  else:
    rho_tor = IMAS_data.profiles_1d.rho_tor
    if not rho_tor:
      if B_0 is None or not IMAS_data.profiles_1d.phi:
        raise ValueError(
            "rho_tor not provided and cannot be calculated from given"
            " equilibrium IDS"
        )
      rho_tor = np.sqrt(IMAS_data.profiles_1d.phi / (np.pi * B_0))
    dpsidrhotor = np.gradient(IMAS_data.profiles_1d.psi, rho_tor)

  flux_surf_avg_grad_psi = IMAS_data.profiles_1d.gm7 * dpsidrhotor
  flux_surf_avg_grad_psi2 = IMAS_data.profiles_1d.gm3 * (dpsidrhotor**2)
  flux_surf_avg_grad_psi2_over_R2 = IMAS_data.profiles_1d.gm2 * (dpsidrhotor**2)
  int_dl_over_Bp = dvoldpsi
  flux_surf_avg_1_over_R2 = IMAS_data.profiles_1d.gm1

  # This branching is needed since currently not all equilibrium codes output
  # <1/R>
  if IMAS_data.profiles_1d.gm9:
    flux_surf_avg_1_over_R = IMAS_data.profiles_1d.gm9
  else:
    logging.warning(
        "Flux surface averaged <1/R> profile (gm9) not found;"
        " assuming <1/R> ≈ 1/R_major_profile"
    )
    flux_surf_avg_1_over_R = 1 / R_major_profile

  # jtor in TORAX is defined as the flux surface average equivalent to the
  # flux-surface current density profile. i.e.
  # jtor_torax \equiv dI/dS = dI/drhon / (dS/drhon) = dI/drhon / spr
  # spr = vpr * <1/R> / ( 2 * np.pi )
  # -> Ip_profile = integrate(y = spr * jtor, x= rhon, initial = 0.0)
  # Validate j_phi exists and is non-empty before using it
  if not IMAS_data.profiles_1d.j_phi or not np.any(IMAS_data.profiles_1d.j_phi):
    raise ValueError(
        "Missing required IMAS profile: profiles_1d.j_phi. "
        "Ensure the equilibrium IDS includes j_phi (e.g. in JETTO outputs) "
        "before loading into TORAX."
    )
  jtor = -1 * IMAS_data.profiles_1d.j_phi
  rhon = IMAS_data.profiles_1d.rho_tor_norm
  if not rhon:
    if B_0 is None or not IMAS_data.profiles_1d.phi:
      raise ValueError(
          "rho_tor_norm not provided and cannot be calculated from given"
          " equilibrium IDS"
      )
    rho_tor = np.sqrt(IMAS_data.profiles_1d.phi / (np.pi * B_0))
    rhon = rho_tor / rho_tor[-1]
  vpr = 4 * np.pi * phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)
  spr = vpr * flux_surf_avg_1_over_R / (2 * np.pi)

  # This Ip_profile by integration results in a minor discrepancy between this
  # term and the total Ip from IDS. ~0.1% for the standard test case.
  Ip_profile_unscaled = scipy.integrate.cumulative_trapezoid(
      y=spr * jtor, x=rhon, initial=0.0
  )

  # Because of the minor discrepancy between Ip_profile[-1] (computed by
  # integration) and global_quantities.ip, here we will scale Ip_profile such
  # that the total plasma current is fully consistent.
  Ip_total = -1 * IMAS_data.global_quantities.ip
  Ip_profile = Ip_profile_unscaled * (Ip_total / Ip_profile_unscaled[-1])

  z_magnetic_axis = np.asarray(IMAS_data.global_quantities.magnetic_axis.z)

  # TODO(b/446608829): Add support for edge geometries from IMAS.

  return {
      "Ip_from_parameters": Ip_from_parameters,
      "R_major": R_major,
      "a_minor": np.asarray(IMAS_data.boundary.minor_radius),
      "B_0": B_0,
      "psi": psi,
      "Ip_profile": Ip_profile,
      "Phi": phi,
      "R_in": R_in,
      "R_out": R_out,
      "F": F,
      "int_dl_over_Bp": int_dl_over_Bp,
      "flux_surf_avg_1_over_R": flux_surf_avg_1_over_R,
      "flux_surf_avg_1_over_R2": flux_surf_avg_1_over_R2,
      "flux_surf_avg_grad_psi": flux_surf_avg_grad_psi,
      "flux_surf_avg_grad_psi2": flux_surf_avg_grad_psi2,
      "flux_surf_avg_grad_psi2_over_R2": flux_surf_avg_grad_psi2_over_R2,
      "flux_surf_avg_B2": IMAS_data.profiles_1d.gm5,
      "flux_surf_avg_1_over_B2": IMAS_data.profiles_1d.gm4,
      "delta_upper_face": IMAS_data.profiles_1d.triangularity_upper,
      "delta_lower_face": IMAS_data.profiles_1d.triangularity_lower,
      "elongation": IMAS_data.profiles_1d.elongation,
      "vpr": vpr,
      "face_centers": face_centers,
      "hires_factor": hires_factor,
      "z_magnetic_axis": z_magnetic_axis,
      "diverted": np.bool(IMAS_data.boundary.type),
      "connection_length_target": None,
      "connection_length_divertor": None,
      "angle_of_incidence_target": None,
      "R_OMP": None,
      "R_target": None,
      "B_pol_OMP": None,
  }


def geometry_from_IMAS(
    face_centers: np.ndarray,
    geometry_directory: str | None = None,
    Ip_from_parameters: bool = False,
    hires_factor: int = 4,
    equilibrium_object: ids_toplevel.IDSToplevel | None = None,
    imas_uri: str | None = None,
    imas_filepath: str | None = None,
    explicit_convert: bool = False,
) -> Mapping[float, dict[str, Any]]:
  """Constructs geometry intermediates for all time slices in an IMAS IDS.

  Currently written for COCOSv17 and DDv4.

  Args:
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    geometry_directory: Directory where to find the equilibrium object. If None,
      it defaults to another dir. See `load_geo_data` implementation.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    equilibrium_object: The equilibrium IDS containing the relevant data.
    imas_uri: The IMAS uri containing the equilibrium data.
    imas_filepath: The path to the IMAS netCDF file containing the equilibrium
      data.
    explicit_convert: Whether to explicitly convert the IDS to the current DD
      version. If True, an explicit conversion will be attempted. Explicit
      conversion is recommended when converting between major DD versions.
      https://imas-python.readthedocs.io/en/latest/multi-dd.html#conversion-of-idss-between-dd-versions

  Returns:
    A mapping from times to dicts of intermediate geometry values, one per
    time slice in the IDS.
  """
  equilibrium = _load_equilibrium(
      equilibrium_object=equilibrium_object,
      imas_uri=imas_uri,
      imas_filepath=imas_filepath,
      geometry_directory=geometry_directory,
      explicit_convert=explicit_convert,
  )

  n_slices = len(equilibrium.time_slice)
  times = np.asarray(equilibrium.time)

  intermediates = {}
  for idx in range(n_slices):
    t = float(times[idx])
    intermediates[t] = _geometry_from_single_slice(
        equilibrium=equilibrium,
        slice_index=idx,
        face_centers=face_centers,
        Ip_from_parameters=Ip_from_parameters,
        hires_factor=hires_factor,
    )

  return intermediates


def geometry_from_single_IMAS_slice(
    face_centers: np.ndarray,
    geometry_directory: str | None = None,
    Ip_from_parameters: bool = False,
    hires_factor: int = 4,
    equilibrium_object: ids_toplevel.IDSToplevel | None = None,
    imas_uri: str | None = None,
    imas_filepath: str | None = None,
    explicit_convert: bool = False,
    slice_index: int = 0,
    slice_time: float | None = None,
) -> dict[str, Any]:
  """Constructs geometry intermediates for a single time slice in an IMAS IDS.

  Currently written for COCOSv17 and DDv4.

  This is the single-slice counterpart to ``geometry_from_IMAS``. It loads
  only one time slice and returns the intermediate geometry values for building
  a ``StandardGeometry``. This enables IMAS geometry to be used in the same
  way as other single-file geometry types (e.g. CHEASE, EQDSK), including as
  entries in a time-dependent ``geometry_configs`` dict.

  The slice to load is selected by one of two mutually exclusive options:

  - ``slice_time``: The IDS time slice whose time is closest to this value
    (in seconds) is selected. Takes precedence over ``slice_index`` when set.
  - ``slice_index``: Integer index into the IDS time slice array. Defaults to
    0 (the first time slice).

  Args:
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    geometry_directory: Directory where to find the equilibrium object. If None,
      it defaults to another dir. See `load_geo_data` implementation.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    hires_factor: Grid refinement factor for poloidal flux <-> plasma current
      calculations.
    equilibrium_object: The equilibrium IDS containing the relevant data.
    imas_uri: The IMAS uri containing the equilibrium data.
    imas_filepath: The path to the IMAS netCDF file containing the equilibrium
      data.
    explicit_convert: Whether to explicitly convert the IDS to the current DD
      version. If True, an explicit conversion will be attempted. Explicit
      conversion is recommended when converting between major DD versions.
      https://imas-python.readthedocs.io/en/latest/multi-dd.html#conversion-of-idss-between-dd-versions
    slice_index: Index of the time slice to load. Defaults to 0 (the first time
      slice). Ignored when ``slice_time`` is provided.
    slice_time: Time (in seconds) of the IDS time slice to load. The slice whose
      time is closest to this value is selected. When provided, takes precedence
      over ``slice_index``.

  Returns:
    A dict of intermediate geometry values for building a StandardGeometry,
    corresponding to the selected time slice.

  Raises:
    ValueError: If `slice_index` is out of range for the number of time slices
      in the IDS.
  """
  equilibrium = _load_equilibrium(
      equilibrium_object=equilibrium_object,
      imas_uri=imas_uri,
      imas_filepath=imas_filepath,
      geometry_directory=geometry_directory,
      explicit_convert=explicit_convert,
  )

  n_slices = len(equilibrium.time_slice)

  if slice_time is not None:
    times = np.asarray(equilibrium.time)
    slice_index = int(np.argmin(np.abs(times - slice_time)))

  if slice_index < 0 or slice_index >= n_slices:
    raise ValueError(
        f"slice_index={slice_index} is out of range for IDS with "
        f"{n_slices} time slice(s)."
    )

  return _geometry_from_single_slice(
      equilibrium=equilibrium,
      slice_index=slice_index,
      face_centers=face_centers,
      Ip_from_parameters=Ip_from_parameters,
      hires_factor=hires_factor,
  )
