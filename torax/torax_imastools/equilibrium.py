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
from typing import Any, Dict

import numpy as np
import scipy

try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any
from torax import constants
from torax.geometry import geometry_loader
from torax.torax_imastools.util import requires_module, face_to_cell


@requires_module("imaspy")
def write_ids_equilibrium_into_config(
    config: dict, equilibrium: IDSToplevel
) -> dict[str, np.ndarray]:
    """Loads the equilibrium into the geometry config.
    Args:
    config: TORAX config object.
    equilibrium: equilibrium IDS to put into the config as the
      equilibrium_object.

    Returns:
    Full config object will the IDS inside."""
    config["geometry"]["geometry_type"] = "imas"
    config["geometry"]["equilibrium_object"] = equilibrium
    return config


@requires_module("imaspy")
def geometry_from_IMAS(
    equilibrium_object: str | IDSToplevel,
    geometry_dir: str | None = None,
    Ip_from_parameters: bool = False,
    n_rho: int = 25,
    hires_fac: int = 4,
) -> Dict:
    """Constructs a StandardGeometryIntermediates from a IMAS equilibrium IDS.
    Args:
      equilibrium_object: Either directly the equilbrium IDS containing the
        relevant data, or the name of the IMAS netCDF file containing the
        equilibrium.
      geometry_dir: Directory where to find the equilibrium object.
        If None, uses the environment variable TORAX_GEOMETRY_DIR if
        available. If that variable is not set and geometry_dir is not
        provided, then it defaults to another dir. See `load_geo_data`
        implementation.
      Ip_from_parameters: If True, the Ip is taken from the parameters and the
        values in the Geometry are resacled to match the new Ip.
      n_rho: Radial grid points (num cells)
      hires_fac: Grid refinement factor for poloidal flux <--> plasma current
        calculations.
    Returns:
      A StandardGeometry instance based on the input file. This can then be
      used to build a StandardGeometry by passing to `build_standard_geometry`.
    """
    # If the equilibrium_object is the file name, load the ids from the netCDF.
    if isinstance(equilibrium_object, str):
        equilibrium = geometry_loader.load_geo_data(
            geometry_dir,
            equilibrium_object,
            geometry_loader.GeometrySource.IMAS,
        )
    elif isinstance(equilibrium_object, IDSToplevel):
        equilibrium = equilibrium_object
    else:
        raise ValueError("equilibrium_object must be a string (file path) or an IDS")
    IMAS_data = equilibrium.time_slice[0]
    B0 = np.abs(equilibrium.vacuum_toroidal_field.b0[0]) #Shoudld it be replaced by reference value .time_slice[0].global_quantities.b_field_phi ?
    Rmaj = np.asarray(IMAS_data.boundary.geometric_axis.r) #Shoudld it be replaced by reference value .vacuum_toroidal_field.r0 ?

    # Poloidal flux (switch sign between ddv3 and ddv4)
    # psi = -1 * IMAS_data.profiles_1d.psi #ddv3
    psi = 1 * IMAS_data.profiles_1d.psi  # ddv4

    # toroidal flux
    Phi = -1 * IMAS_data.profiles_1d.phi

    # midplane radii
    Rin = IMAS_data.profiles_1d.r_inboard
    Rout = IMAS_data.profiles_1d.r_outboard
    # toroidal field flux function
    F = -1 * IMAS_data.profiles_1d.f

    # Flux surface integrals of various geometry quantities
    # IDS Contour integrals
    if len(IMAS_data.profiles_1d.dvolume_dpsi) > 0:
        dvoldpsi = 1 * IMAS_data.profiles_1d.dvolume_dpsi #Sign changed ddv4
    else:
        dvoldpsi = (
            1
            * np.gradient(IMAS_data.profiles_1d.volume)
            / np.gradient(IMAS_data.profiles_1d.psi)
        )
    # dpsi_drho_tor (switch sign between ddv3 and ddv4)
    if len(IMAS_data.profiles_1d.dpsi_drho_tor) > 0:
        # dpsidrhotor = -1 * IMAS_data.profiles_1d.dpsi_drho_tor #ddv3
        dpsidrhotor = 1 * IMAS_data.profiles_1d.dpsi_drho_tor  # ddv4
    else:
        # dpsidrhotor = -1 * np.gradient(IMAS_data.profiles_1d.psi) \
        #    / np.gradient(IMAS_data.profiles_1d.rho_tor)           #ddv3
        dpsidrhotor = (
            1
            * np.gradient(IMAS_data.profiles_1d.psi)
            / np.gradient(IMAS_data.profiles_1d.rho_tor)
        )  # ddv4
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
    # spr = vpr / ( 2 * np.pi * Rmaj)
    # -> Ip_profile = integrate(y = spr * jtor, x= rhon, initial = 0.0)
    jtor = -1 * IMAS_data.profiles_1d.j_phi
    rhon = IMAS_data.profiles_1d.rho_tor_norm
    vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)
    spr = vpr / (2 * np.pi * Rmaj)
    Ip_profile = scipy.integrate.cumulative_trapezoid(y=spr * jtor, x=rhon, initial=0.0)

    # To check
    z_magnetic_axis = np.asarray(IMAS_data.global_quantities.magnetic_axis.z)

    return {
        "Ip_from_parameters": Ip_from_parameters,
        "Rmaj": Rmaj,
        "Rmin": np.asarray(IMAS_data.boundary.minor_radius),
        "B": B0,
        "psi": psi,
        "Ip_profile": Ip_profile,
        "Phi": Phi,
        "Rin": Rin,
        "Rout": Rout,
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
        "hires_fac": hires_fac,
        "z_magnetic_axis": z_magnetic_axis,
    }

@requires_module("imaspy")
def geometry_to_IMAS(SimState) -> IDSToplevel:
    """Constructs an IMAS equilibrium IDS from a StandardGeometry object.
    Takes the cell grid as a basis and converts values on face grid to cell.
    Args:
      SimState: A ToraxSimState object containing:
        geometry: TORAX StandardGeometry object.
        core_profiles: TORAX core_profiles for q profile.
        post_processed_outputs: TORAX post_processed_outputs containing useful
          variables for coupling with equilibrium code such as p' and FF'.
    Returns:
      equilibrium IDS based on the current TORAX sim.State object.
    """

    geometry = SimState.geometry
    core_profiles = SimState.core_profiles
    post_processed_outputs = SimState.post_processed_outputs
    #Rebuilding the equilibrium from the geometry object (Which should remain unchanged by the transport code), is it needed or do we only need coupling variables ?
    equilibrium = imaspy.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = 1 #Should be 0 or 1 ?
    equilibrium.ids_properties.comment = "equilibrium IDS built from ToraxSimState object."
    equilibrium.time.resize(1)
    equilibrium.time = [SimState.t] #What time should be set ? Needed for B0
    equilibrium.vacuum_toroidal_field.b0 = -1 * geometry.B0
    equilibrium.time_slice.resize(1)
    eq = equilibrium.time_slice[0]
    eq.boundary.geometric_axis.r = geometry.Rmaj
    eq.boundary.minor_radius = geometry.Rmin
    # determine sign how?
    eq.profiles_1d.psi = core_profiles.psi.value.copy()
    # determine sign how?
    eq.profiles_1d.phi = -1 * geometry.Phi
    eq.profiles_1d.r_inboard = geometry.Rin
    eq.profiles_1d.r_outboard = geometry.Rout

    eq.profiles_1d.triangularity_upper = face_to_cell(geometry.delta_upper_face)
    eq.profiles_1d.triangularity_lower = face_to_cell(geometry.delta_lower_face)
    eq.profiles_1d.elongation = geometry.elongation
    eq.global_quantities.magnetic_axis.z = geometry.z_magnetic_axis

    eq.profiles_1d.j_phi = -1 * geometry.jtot
    eq.profiles_1d.volume = geometry.volume
    eq.profiles_1d.area = geometry.area
    eq.profiles_1d.rho_tor = np.sqrt(geometry.Phi / (np.pi * geometry.B0))
    eq.profiles_1d.rho_tor_norm = geometry.torax_mesh.cell_centers

    dvoldpsi = (
          1
          * np.gradient(eq.profiles_1d.volume)
          / np.gradient(eq.profiles_1d.psi)
      )
    dpsidrhotor = (
        1
        * np.gradient(eq.profiles_1d.psi)
        / np.gradient(eq.profiles_1d.rho_tor)
    )
    eq.profiles_1d.dvolume_dpsi = dvoldpsi
    eq.profiles_1d.dpsi_drho_tor = dpsidrhotor
    eq.profiles_1d.gm1 = geometry.g3
    eq.profiles_1d.gm7 = geometry.g0/(dvoldpsi * dpsidrhotor)
    eq.profiles_1d.gm3 = geometry.g1 / (dpsidrhotor ** 2 * dvoldpsi**2)
    eq.profiles_1d.gm2 = geometry.g2 / (dpsidrhotor ** 2 * dvoldpsi**2)

    #Quantities computed by the transport code useful for coupling with equilibrium code
    eq.profiles_1d.pressure = face_to_cell(post_processed_outputs.pressure_thermal_tot_face)
    eq.profiles_1d.dpressure_dpsi = face_to_cell(post_processed_outputs.pprime_face)

    #<j.B>/B0, could be useful to calculate and use instead of FF' (Formula not checked, has to be tested and verified)
    # determine sign how?
    eq.profiles_1d.f = -1 * geometry.F #Is probably not self-consistent due to the evolution of the state by the solver.
    eq.profiles_1d.f_df_dpsi = face_to_cell(post_processed_outputs.FFprime_face)
    eq.profiles_1d.q = face_to_cell(core_profiles.q_face)

    return equilibrium
