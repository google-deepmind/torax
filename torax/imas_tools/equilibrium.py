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

"""Useful functions for handling of IMAS IDSs and converts them into TORAX
objects"""
from typing import Any, Optional, TYPE_CHECKING

import imas
from imas.ids_toplevel import IDSToplevel
import numpy as np
import scipy

from torax._src.geometry import geometry_loader

# Imports added for type hinting,
# TYPE_CHECKING guard to prevent circular import at runtime
if TYPE_CHECKING:
  from torax._src.orchestration import sim_state
  from torax._src.output_tools import post_processing


def write_ids_equilibrium_into_config(
    config: dict, equilibrium: IDSToplevel
) -> dict[str, np.ndarray]:
  """Loads the equilibrium into the geometry config.
  Args:
  config: TORAX config object.
  equilibrium: equilibrium IDS to put into the config as the
    equilibrium_object.

  Returns:
  Full IMASconfig object for the geometry with the IDS inside."""
  config["geometry"]["geometry_type"] = "imas"
  config["geometry"]["equilibrium_object"] = equilibrium
  return config


def geometry_from_IMAS(
    geometry_directory: str | None = None,
    Ip_from_parameters: bool = False,
    n_rho: int = 25,
    hires_factor: int = 4,
    equilibrium_object: Optional[IDSToplevel] = None,
    imas_uri: Optional[str] = None,
    imas_filepath: Optional[str] = None,
) -> dict[str, Any]:
  """Constructs a StandardGeometryIntermediates from a IMAS equilibrium IDS.
  Args:
    geometry_directory: Directory where to find the equilibrium object.
      If None, it defaults to another dir. See `load_geo_data`
      implementation.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are resacled to match the new Ip.
    n_rho: Radial grid points (num cells)
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    equilibrium_object: The equilibrium IDS containing the relevant data.
    imas_uri: The IMAS uri containing the equilibrium data.
    imas_filepath: The path to the IMAS netCDF file containing the equilibrium
      data.
  Returns:
    A StandardGeometry instance based on the input file. This can then be
    used to build a StandardGeometry by passing to `build_standard_geometry`.
  """
  # If the equilibrium_object is the file name, load the ids from the netCDF.
  if equilibrium_object is not None:
    equilibrium = equilibrium_object
  elif imas_uri is not None:
    equilibrium = geometry_loader.load_geo_data(
        geometry_directory,
        imas_uri,
        geometry_loader.GeometrySource.IMAS,
    )
  elif imas_filepath is not None:
    equilibrium = geometry_loader.load_geo_data(
        geometry_directory,
        imas_filepath,
        geometry_loader.GeometrySource.IMAS,
    )
  else:
    raise ValueError(
        "equilibrium_object must be a string (file path) or an IDS"
    )
  IMAS_data = equilibrium.time_slice[0]
  B_0 = np.abs(
      equilibrium.vacuum_toroidal_field.b0[0]
  )  # Should it be replaced by .time_slice[0].global_quantities.b_field_phi ?
  R_major = np.asarray(
      equilibrium.vacuum_toroidal_field.r0
  )  # Should it be replaced by IMAS_data.boundary.geometric_axis.r ?

  # Poloidal flux
  psi = 1 * IMAS_data.profiles_1d.psi  # Sign changed ddv4

  # toroidal flux
  phi = -1 * IMAS_data.profiles_1d.phi

  # midplane radii
  R_in = IMAS_data.profiles_1d.r_inboard
  R_out = IMAS_data.profiles_1d.r_outboard
  # toroidal field flux function
  F = -1 * IMAS_data.profiles_1d.f

  # Flux surface integrals of various geometry quantities
  # IDS Contour integrals
  if len(IMAS_data.profiles_1d.dvolume_dpsi) > 0:
    dvoldpsi = 1 * IMAS_data.profiles_1d.dvolume_dpsi  # Sign changed ddv4
  else:
    dvoldpsi = np.gradient(
        IMAS_data.profiles_1d.volume, IMAS_data.profiles_1d.psi
    )
  # dpsi_drho_tor
  if len(IMAS_data.profiles_1d.dpsi_drho_tor) > 0:
    dpsidrhotor = 1 * IMAS_data.profiles_1d.dpsi_drho_tor  # Sign  changed ddv4
  else:
    rho_tor = IMAS_data.profiles_1d.rho_tor
    if len(rho_tor) == 0:
      if B_0 is None or len(IMAS_data.profiles_1d.phi) == 0:
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

  # jtor = dI/drhon / (drho/dS) = dI/drhon / spr
  # spr = vpr / ( 2 * np.pi * R_major)
  # -> Ip_profile = integrate(y = spr * jtor, x= rhon, initial = 0.0)
  jtor = -1 * IMAS_data.profiles_1d.j_phi
  rhon = IMAS_data.profiles_1d.rho_tor_norm
  if len(rhon) == 0:
    if B_0 is None or len(IMAS_data.profiles_1d.phi) == 0:
      raise ValueError(
          "rho_tor_norm not provided and cannot be calculated from given"
          " equilibrium IDS"
      )
    rho_tor = np.sqrt(IMAS_data.profiles_1d.phi / (np.pi * B_0))
    rhon = rho_tor / rho_tor[-1]
  vpr = 4 * np.pi * phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)
  spr = vpr / (2 * np.pi * R_major)
  # this Ip_profile by integration results in a discrepancy between this term
  # and the total ip from IDS
  Ip_profile_unscaled = scipy.integrate.cumulative_trapezoid(
      y=spr * jtor, x=rhon, initial=0.0
  )

  # Because of the discrepancy between Ip_profile[-1] (computed by integration)
  # and global_quantities.ip, here we will scale Ip_profile such that the total
  # plasma current is equal
  Ip_total = -1 * IMAS_data.global_quantities.ip
  Ip_profile = Ip_profile_unscaled * (
      Ip_total / Ip_profile_unscaled[-1]
  )  # scaled Ip profile such that the total plasma current is consistent

  # To check
  z_magnetic_axis = np.asarray(IMAS_data.global_quantities.magnetic_axis.z)

  return {
      "Ip_from_parameters": Ip_from_parameters,
      "R_major": R_major,
      "a_minor": np.asarray(IMAS_data.boundary.minor_radius),
      "B_0": B_0,
      "psi": psi,
      "Ip_profile": Ip_profile,
      "phi": phi,
      "R_in": R_in,
      "R_out": R_out,
      "F": F,
      "int_dl_over_Bp": int_dl_over_Bp,
      "flux_surf_avg_1_over_R2": flux_surf_avg_1_over_R2,
      "flux_surf_avg_RBp": flux_surf_avg_RBp,
      "flux_surf_avg_R2Bp2": flux_surf_avg_R2Bp2,
      "flux_surf_avg_Bp2": flux_surf_avg_Bp2,
      "delta_upper_face": IMAS_data.profiles_1d.triangularity_upper,
      "delta_lower_face": IMAS_data.profiles_1d.triangularity_lower,
      "elongation": IMAS_data.profiles_1d.elongation,
      "vpr": vpr,
      "n_rho": n_rho,
      "hires_factor": hires_factor,
      "z_magnetic_axis": z_magnetic_axis,
  }


def geometry_to_IMAS(
    sim_state: "sim_state.ToraxSimState",
    post_processed_outputs: "post_processing.PostProcessedOutputs",
    equilibrium_in: IDSToplevel | None = None,
) -> IDSToplevel:
  """Constructs an IMAS equilibrium IDS from a StandardGeometry object.
  Takes the cell grid as a basis and converts values on face grid to cell.
  Args:
    sim_state: A ToraxSimState object containing:
      - geometry: TORAX StandardGeometry object.
      - core_profiles: TORAX core_profiles for q profile.
      - post_processed_outputs: TORAX post_processed_outputs containing useful
        variables for coupling with equilibrium code such as p' and FF'.
    equilibrium_in: Optional equilibrium IDS to specify the plasma boundary
        which is not stored in TORAX variables but needed for coupling with
        NICE for example.
  Returns:
    equilibrium IDS based on the current TORAX sim.State object.
  """

  geometry = sim_state.geometry
  core_profiles = sim_state.core_profiles
  # Rebuilding the equilibrium from the geometry object
  # (Which should remain unchanged by the transport code),
  # is it needed or do we only need coupling variables ?
  equilibrium = imas.IDSFactory().equilibrium()
  equilibrium.ids_properties.homogeneous_time = 1  # Should be 0 or 1 ?
  equilibrium.ids_properties.comment = (
      "equilibrium IDS built from ToraxSimState object."
  )
  equilibrium.time.resize(1)
  equilibrium.time = [sim_state.t]  # What time should be set ? Needed for B_0
  equilibrium.vacuum_toroidal_field.r0 = geometry.R_major
  equilibrium.vacuum_toroidal_field.b0.resize(1)
  equilibrium.vacuum_toroidal_field.b0[0] = -1 * geometry.B_0
  equilibrium.time_slice.resize(1)
  eq = equilibrium.time_slice[0]
  eq.boundary.geometric_axis.r = geometry.R_major
  eq.boundary.minor_radius = geometry.a_minor
  # determine sign how?
  eq.profiles_1d.psi = core_profiles.psi.face_value()
  # determine sign how?
  eq.profiles_1d.phi = -1 * geometry.phi_face
  eq.profiles_1d.r_inboard = geometry.R_in_face
  eq.profiles_1d.r_outboard = geometry.R_out_face

  eq.profiles_1d.triangularity_upper = geometry.delta_upper_face
  eq.profiles_1d.triangularity_lower = geometry.delta_lower_face
  eq.profiles_1d.elongation = geometry.elongation_face
  eq.global_quantities.magnetic_axis.z = geometry.z_magnetic_axis
  eq.global_quantities.ip = -1 * geometry.Ip_profile_face[-1]
  eq.profiles_1d.j_phi = -1 * core_profiles.j_total_face
  eq.profiles_1d.volume = geometry.volume_face
  eq.profiles_1d.area = geometry.area_face
  eq.profiles_1d.rho_tor = np.sqrt(geometry.phi_face / (np.pi * geometry.B_0))
  eq.profiles_1d.rho_tor_norm = geometry.torax_mesh.face_centers

  dvoldpsi = (
      1 * np.gradient(eq.profiles_1d.volume) / np.gradient(eq.profiles_1d.psi)
  )
  dpsidrhotor = (
      1 * np.gradient(eq.profiles_1d.psi) / np.gradient(eq.profiles_1d.rho_tor)
  )
  eq.profiles_1d.dvolume_dpsi = dvoldpsi
  eq.profiles_1d.dpsi_drho_tor = dpsidrhotor
  eq.profiles_1d.gm1 = geometry.g3_face
  eq.profiles_1d.gm7 = geometry.g0_face / (dvoldpsi * dpsidrhotor)
  eq.profiles_1d.gm3 = geometry.g1_face / (dpsidrhotor**2 * dvoldpsi**2)
  eq.profiles_1d.gm2 = geometry.g2_face / (dpsidrhotor**2 * dvoldpsi**2)

  # Quantities computed by the transport code useful for coupling with
  # equilibrium code
  eq.profiles_1d.pressure = (
      post_processed_outputs.pressure_thermal_total.face_value()
  )
  eq.profiles_1d.dpressure_dpsi = post_processed_outputs.pprime

  # <j.B>/B_0, could be useful to calculate and use instead of FF'
  # determine sign how? Is probably not self-consistent due to the
  # evolution of the state by the solver.
  eq.profiles_1d.f = -1 * geometry.F_face
  eq.profiles_1d.f_df_dpsi = post_processed_outputs.FFprime
  eq.profiles_1d.q = core_profiles.q_face

  # Optionally maps fixed quantities not evolved by TORAX and read directly
  # from input equilibrium. Needed to couple with NICE inverse
  if equilibrium_in is not None:
    eq.boundary.outline.r = equilibrium_in.time_slice[0].boundary.outline.r
    eq.boundary.outline.z = equilibrium_in.time_slice[0].boundary.outline.z

  return equilibrium
