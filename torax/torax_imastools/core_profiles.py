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
from typing import Any

try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any

from torax.state import ToraxSimState
from torax.torax_imastools.util import face_to_cell, requires_module


@requires_module("imaspy")
def core_profiles_to_IMAS(
    ids: IDSToplevel, state: ToraxSimState, geometry
) -> IDSToplevel:
    """Converts torax core_profiles to IMAS IDS.
    Takes the cell grid as a basis and converts values on face grid to cell.
    Args:
    ids: IDS object
    state: torax state object

    Returns:
    filled IDS object"""
    t = state.t
    cp_state = state.core_profiles
    ids.ids_properties.comment = "Grid based on torax cell grid, used cell grid values and interpolated face grid values"
    ids.ids_properties.homogeneous_time = 1
    ids.time = [t]
    ids.vacuum_toroidal_field.b0.resize(1)
    ids.global_quantities.current_non_inductive.resize(1)
    ids.profiles_1d.resize(1)
    ids.profiles_1d[0].ion.resize(2)
    ids.profiles_1d[0].ion[0].element.resize(1)
    ids.profiles_1d[0].ion[1].element.resize(1)
    ids.vacuum_toroidal_field.r0 = geometry.Rmaj
    ids.vacuum_toroidal_field.b0[0] = geometry.B0
    ids.global_quantities.current_non_inductive[0] = cp_state.currents.I_bootstrap
    ids.profiles_1d[0].grid.rho_tor_norm = geometry.rho_norm
    ids.profiles_1d[0].grid.rho_tor = geometry.rho
    ids.profiles_1d[0].grid.psi = cp_state.psi.value
    ids.profiles_1d[0].grid.volume = geometry.volume
    ids.profiles_1d[0].grid.area = geometry.area
    ids.profiles_1d[0].electrons.temperature = cp_state.temp_el.value
    ids.profiles_1d[0].electrons.density = cp_state.ne.value
    ids.profiles_1d[0].ion[0].z_ion = cp_state.Zi
    ids.profiles_1d[0].ion[0].temperature = cp_state.temp_ion.value
    ids.profiles_1d[0].ion[0].density = cp_state.ni.value
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[0].element[0].a = cp_state.Zi
    ids.profiles_1d[0].ion[0].element[0].z_n = cp_state.Ai
    ids.profiles_1d[0].ion[1].z_ion = cp_state.Zimp
    ids.profiles_1d[0].ion[1].temperature = cp_state.temp_ion.value
    ids.profiles_1d[0].ion[1].density = cp_state.nimp.value
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[1].element[0].a = cp_state.Zimp
    ids.profiles_1d[0].ion[1].element[0].z_n = cp_state.Aimp
    ids.profiles_1d[0].q = face_to_cell(cp_state.q_face)
    ids.profiles_1d[0].magnetic_shear = face_to_cell(cp_state.s_face)
    ids.profiles_1d[0].j_total = cp_state.currents.jtot
    ids.profiles_1d[0].j_ohmic = cp_state.currents.johm
    ids.profiles_1d[0].j_non_inductive = cp_state.currents.external_current_source
    ids.profiles_1d[0].j_bootstrap = cp_state.currents.j_bootstrap
    ids.profiles_1d[0].conductivity_parallel = cp_state.currents.sigma
    # ids. = cp_state.psidot.value
    # ids. = face_to_cell(cp_state.currents.jtot_face)
    # ids. = face_to_cell(cp_state.currents.Ip_profile_face)
    # ids. = face_to_cell(cp_state.currents.j_bootstrap_face)
    return ids
