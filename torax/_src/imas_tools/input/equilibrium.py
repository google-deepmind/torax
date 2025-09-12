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
import logging
import os
from typing import Any

import imas
from imas import ids_toplevel
import numpy as np
import scipy
from torax._src.geometry import geometry_loader


# pylint: disable=invalid-name
def geometry_from_IMAS(
    geometry_directory: str | None = None,
    Ip_from_parameters: bool = False,
    n_rho: int = 25,
    hires_factor: int = 4,
    slice_time: float | None = None,
    slice_index: int = 0,
    equilibrium_object: ids_toplevel.IDSToplevel | None = None,
    imas_uri: str | None = None,
    imas_filepath: str | None = None,
) -> dict[str, Any]:
  """Constructs a StandardGeometryIntermediates from a IMAS equilibrium IDS.

  Currently written for COCOSv17 and DDv4.

  Args:
    geometry_directory: Directory where to find the equilibrium object. If None,
      it defaults to another dir. See `load_geo_data` implementation.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    n_rho: Radial grid points (num cells)
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    slice_time: Time of slice to load from IMAS IDS. If given, overrides
      slice_index.
    slice_index: Index of slice to load from IMAS IDS.
    equilibrium_object: The equilibrium IDS containing the relevant data.
    imas_uri: The IMAS uri containing the equilibrium data.
    imas_filepath: The path to the IMAS netCDF file containing the equilibrium
      data.

  Returns:
    A StandardGeometry instance based on the input file. This can then be
    used to build a StandardGeometry by passing to `build_standard_geometry`.

  Raises:
    IndexError: If `slice_index` is out of bounds for the available time slices
      in the equilibrium object.
  """
  # If the equilibrium_object is the file name, load the ids from the netCDF.
  if equilibrium_object is not None:
    equilibrium = equilibrium_object
  elif imas_uri is not None:
    equilibrium = _load_geo_data(
        imas_uri,
        geometry_directory,
    )
  elif imas_filepath is not None:
    equilibrium = _load_geo_data(
        imas_filepath,
        geometry_directory,
    )
  else:
    raise ValueError(
        "equilibrium_object must be a string (file path) or an IDS"
    )
  # TODO(b/431977390): Currently only a single time slice is used, extend to
  # support multiple time slices.
  # Convert time to index
  if slice_time is not None:
    if not np.all(equilibrium.time[:-1] <= equilibrium.time[1:]):
      sorting_indices = np.argsort(equilibrium.time)
    else:
      sorting_indices = np.arange(len(equilibrium.time))
    # Find the closest time in the IDS that is <= slice_time
    slice_index = (
        np.searchsorted(
            equilibrium.time[sorting_indices], slice_time, side="right"
        )
        - 1
    )
    if not np.allclose(
        equilibrium.time[sorting_indices][slice_index], slice_time, atol=1e-9
    ):
      logging.warning(
          "Requested t=%s not in IDS; using t=%s)",
          slice_time,
          equilibrium.time[slice_index],
      )

  if slice_index >= len(equilibrium.time_slice):
    raise IndexError(
        f"slice_index={slice_index} out of range for IDS with "
        f"{len(equilibrium.time_slice)} time slices"
    )
  IMAS_data = equilibrium.time_slice[slice_index]
  R_major = np.asarray(equilibrium.vacuum_toroidal_field.r0)
  B_0 = np.asarray(np.abs(equilibrium.vacuum_toroidal_field.b0[0]))

  # Poloidal flux.
  psi = 1 * IMAS_data.profiles_1d.psi  # Sign changed ddv4

  # Toroidal flux.
  phi = -1 * IMAS_data.profiles_1d.phi

  # Midplane radii.
  R_in = IMAS_data.profiles_1d.r_inboard
  R_out = IMAS_data.profiles_1d.r_outboard
  # toroidal field flux function
  F = -1 * IMAS_data.profiles_1d.f

  # Flux surface integrals of various geometry quantities.
  # IDS Contour integrals.
  if IMAS_data.profiles_1d.dvolume_dpsi:
    dvoldpsi = 1 * IMAS_data.profiles_1d.dvolume_dpsi  # Sign changed ddv4.
  else:
    dvoldpsi = np.gradient(
        IMAS_data.profiles_1d.volume, IMAS_data.profiles_1d.psi
    )
  # dpsi_drho_tor
  if IMAS_data.profiles_1d.dpsi_drho_tor:
    dpsidrhotor = 1 * IMAS_data.profiles_1d.dpsi_drho_tor  # Sign changed ddv4.
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

  flux_surf_avg_RBp = (
      IMAS_data.profiles_1d.gm7 * dpsidrhotor / (2 * np.pi)
  )  # dpsi, C0/C1
  flux_surf_avg_R2Bp2 = (
      IMAS_data.profiles_1d.gm3 * (dpsidrhotor**2) / (4 * np.pi**2)
  )  # C4/C1
  flux_surf_avg_Bp2 = (
      IMAS_data.profiles_1d.gm2 * (dpsidrhotor**2) / (4 * np.pi**2)
  )  # C3/C1
  int_dl_over_Bp = dvoldpsi  # C1
  flux_surf_avg_1_over_R2 = IMAS_data.profiles_1d.gm1  # C2/C1

  # This branching is needed since currently not all equilibrium codes output
  # <1/R>
  if IMAS_data.profiles_1d.gm9:
    flux_surf_avg_1_over_R = IMAS_data.profiles_1d.gm9
  else:
    logging.warning(
        "Flux surface averaged <1/R> profile (gm9) not found;"
        " assuming <1/R> ≈ 1/R_major (constant)"
    )
    flux_surf_avg_1_over_R = 1 / R_major

  # jtor in TORAX is defined as the flux surface average equivalent to the
  # flux-surface current density profile. i.e.
  # jtor_torax \equiv dI/dS = dI/drhon / (dS/drhon) = dI/drhon / spr
  # spr = vpr / ( 2 * np.pi * R_major)
  # -> Ip_profile = integrate(y = spr * jtor, x= rhon, initial = 0.0)
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
  spr = vpr / (2 * np.pi * R_major)
  # This Ip_profile by integration results in a discrepancy between this term
  # and the total Ip from IDS.
  Ip_profile_unscaled = scipy.integrate.cumulative_trapezoid(
      y=spr * jtor, x=rhon, initial=0.0
  )

  # Because of the discrepancy between Ip_profile[-1] (computed by integration)
  # and global_quantities.ip, here we will scale Ip_profile such that the total
  # plasma current is equal.
  Ip_total = -1 * IMAS_data.global_quantities.ip
  Ip_profile = Ip_profile_unscaled * (Ip_total / Ip_profile_unscaled[-1])

  z_magnetic_axis = np.asarray(IMAS_data.global_quantities.magnetic_axis.z)

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
      "flux_surf_avg_RBp": flux_surf_avg_RBp,
      "flux_surf_avg_R2Bp2": flux_surf_avg_R2Bp2,
      "flux_surf_avg_Bp2": flux_surf_avg_Bp2,
      "flux_surf_avg_B2": IMAS_data.profiles_1d.gm5,
      "flux_surf_avg_1_over_B2": IMAS_data.profiles_1d.gm4,
      "delta_upper_face": IMAS_data.profiles_1d.triangularity_upper,
      "delta_lower_face": IMAS_data.profiles_1d.triangularity_lower,
      "elongation": IMAS_data.profiles_1d.elongation,
      "vpr": vpr,
      "n_rho": n_rho,
      "hires_factor": hires_factor,
      "z_magnetic_axis": z_magnetic_axis,
  }


def _load_geo_data(
    uri: str,
    geometry_directory: str | None = None,
) -> ids_toplevel.IDSToplevel:
  """Loads a full equilibrium IDS for a given uri or path_name and a given ids_name."""
  geometry_directory = geometry_loader.get_geometry_dir(geometry_directory)
  uri = os.path.join(geometry_directory, uri)
  with imas.DBEntry(uri=uri, mode="r") as db:
    ids = db.get(ids_name="equilibrium")
  return ids
