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
"""Helper functions for IMAS output."""

import imas
from imas import ids_toplevel
import numpy as np
from torax._src.geometry import standard_geometry
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.output_tools import post_processing


# pylint: disable=invalid-name
def torax_state_to_imas_equilibrium(
    sim_state: sim_state_lib.ToraxSimState,
    post_processed_outputs: post_processing.PostProcessedOutputs,
    equilibrium_in: ids_toplevel.IDSToplevel | None = None,
) -> ids_toplevel.IDSToplevel:
  """Constructs an IMAS equilibrium IDS from TORAX state quantities.

  Takes the cell grid as a basis and converts values on face grid to cell.

  Args:
    sim_state: A ToraxSimState object.
    post_processed_outputs: TORAX post_processed_outputs containing useful
      variables for coupling with equilibrium code such as p' and FF'.
    equilibrium_in: Optional equilibrium IDS to specify the plasma boundary
      which is not stored in TORAX variables but needed for coupling with NICE
      for example.

  Returns:
    Equilibrium IDS based on the current TORAX simulation state object.
  """

  geometry = sim_state.geometry
  if not isinstance(geometry, standard_geometry.StandardGeometry):
    raise ValueError("geometry_to_IMAS only supports StandardGeometry objects.")
  core_profiles = sim_state.core_profiles
  # Rebuilding the equilibrium from the geometry object
  equilibrium = imas.IDSFactory().equilibrium()
  equilibrium.ids_properties.homogeneous_time = 1
  equilibrium.ids_properties.comment = (
      "equilibrium IDS built from ToraxSimState object."
  )
  equilibrium.time.resize(1)
  equilibrium.time = [sim_state.t]
  equilibrium.vacuum_toroidal_field.r0 = geometry.R_major
  equilibrium.vacuum_toroidal_field.b0.resize(1)
  equilibrium.vacuum_toroidal_field.b0[0] = -1 * geometry.B_0
  equilibrium.time_slice.resize(1)
  eq = equilibrium.time_slice[0]
  eq.boundary.geometric_axis.r = geometry.R_major
  eq.boundary.minor_radius = geometry.a_minor
  eq.profiles_1d.psi = core_profiles.psi.face_value()
  eq.profiles_1d.phi = -1 * geometry.Phi_face
  eq.profiles_1d.r_inboard = geometry.R_in_face
  eq.profiles_1d.r_outboard = geometry.R_out_face

  eq.profiles_1d.triangularity_upper = geometry.delta_upper_face
  eq.profiles_1d.triangularity_lower = geometry.delta_lower_face
  eq.profiles_1d.elongation = geometry.elongation_face
  try:
    eq.global_quantities.magnetic_axis.z = geometry.z_magnetic_axis()
  except ValueError as e:
    pass
  eq.global_quantities.ip = -1 * geometry.Ip_profile_face[-1]
  eq.profiles_1d.j_phi = -1 * core_profiles.j_total_face
  eq.profiles_1d.volume = geometry.volume_face
  eq.profiles_1d.area = geometry.area_face
  eq.profiles_1d.rho_tor = geometry.rho_face
  eq.profiles_1d.rho_tor_norm = geometry.torax_mesh.face_centers

  dvoldpsi = (
      1 * np.gradient(eq.profiles_1d.volume) / np.gradient(eq.profiles_1d.psi)
  )
  dpsidrhotor = (
      1 * np.gradient(eq.profiles_1d.psi) / np.gradient(eq.profiles_1d.rho_tor)
  )
  eq.profiles_1d.dpsi_drho_tor = dpsidrhotor
  eq.profiles_1d.dvolume_dpsi = dvoldpsi
  eq.profiles_1d.gm1 = geometry.g3_face
  # Avoid division by zero for gm calculations by prescribing on-axis value.
  # gm7 = <\nabla V> / (dV/drhotor)
  gm7 = np.array(geometry.g0_face[1:] / (dvoldpsi[1:] * dpsidrhotor[1:]))
  gm7_on_axis = np.array([1.0])
  gm7 = np.concat([gm7_on_axis, gm7])
  eq.profiles_1d.gm7 = gm7
  # gm3 = <(\nabla V)**2>/(dV/drhotor)**2
  gm3 = np.array(
      geometry.g1_face[1:] / (dpsidrhotor[1:] ** 2 * dvoldpsi[1:] ** 2)
  )
  gm3_on_axis = np.array([1.0])
  gm3 = np.concat([gm3_on_axis, gm3])
  eq.profiles_1d.gm3 = gm3
  # gm2 = <(\nabla V)**2/R**2>/(dV/drhotor)**2
  gm2 = np.array(geometry.g2_face[1:] / (dpsidrhotor[1:]**2 * dvoldpsi[1:]**2))
  gm2_on_axis = np.array([1 / (geometry.R_major**2)])
  gm2 = np.concat([gm2_on_axis, gm2])
  eq.profiles_1d.gm2 = gm2

  # Quantities useful for coupling with equilibrium code
  eq.profiles_1d.pressure = (
      post_processed_outputs.pressure_thermal_total.face_value()
  )
  eq.profiles_1d.dpressure_dpsi = post_processed_outputs.pprime

  # <j.B>/B_0, could be useful to calculate and use instead of FF'
  eq.profiles_1d.f = -1 * geometry.F_face
  eq.profiles_1d.f_df_dpsi = post_processed_outputs.FFprime
  eq.profiles_1d.q = core_profiles.q_face

  # Optionally maps fixed quantities not evolved by TORAX and read directly
  # from input equilibrium.
  if equilibrium_in is not None:
    eq.boundary.outline.r = equilibrium_in.time_slice[0].boundary.outline.r
    eq.boundary.outline.z = equilibrium_in.time_slice[0].boundary.outline.z

  return equilibrium
