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

"""Module containing functions for saving and loading simulation output."""
from typing import TypeAlias

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.sources import source_profiles
import xarray as xr

import os


# Core profiles.
TEMP_EL = "temp_el"
TEMP_EL_RIGHT_BC = "temp_el_right_bc"
TEMP_ION = "temp_ion"
TEMP_ION_RIGHT_BC = "temp_ion_right_bc"
PSI = "psi"
PSIDOT = "psidot"
PSI_RIGHT_GRAD_BC = "psi_right_grad_bc"
NE = "ne"
NE_RIGHT_BC = "ne_right_bc"
NI = "ni"
NI_RIGHT_BC = "ni_right_bc"
JTOT = "jtot"
JTOT_FACE = "jtot_face"
JOHM = "johm"
JOHM_FACE = "johm_face"
JEXT = "jext"
JEXT_FACE = "jext_face"
J_BOOTSTRAP = "j_bootstrap"
J_BOOTSTRAP_FACE = "j_bootstrap_face"
I_BOOTSTRAP = "I_bootstrap"
SIGMA = "sigma"
Q_FACE = "q_face"
S_FACE = "s_face"
NREF = "nref"

# Core transport.
CHI_FACE_ION = "chi_face_ion"
CHI_FACE_EL = "chi_face_el"
D_FACE_EL = "d_face_el"
V_FACE_EL = "v_face_el"

# Geometry.
VPR = "vpr"
SPR = "spr"
VPR_FACE = "vpr_face"
SPR_FACE = "spr_face"

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Tuple of (path_to_xarray_file, time_to_load_from).
FilepathAndTime: TypeAlias = tuple[str, float]


def load_state_file(
    filepath_and_time: FilepathAndTime, data_var: str
) -> xr.DataArray:
  """Loads a state file from a filepath."""
  path, t = filepath_and_time
  if os.path.exists(path):
    with open(path, "rb") as f:
      logging.info("Loading %s from state file %s, time %s", data_var, path, t)
      da = xr.load_dataset(f).sel(time=slice(t, None)).data_vars[data_var]
      earliest_time = float(da.coords["time"][0])
      logging.info("Earliest time in file: %.2f", earliest_time)
      # Shift the time coordinate to start at 0.
      da = da.assign_coords({"time": da.coords["time"] - earliest_time})
      if RHO_CELL_NORM in da.coords:
        return da.rename({RHO_CELL_NORM: interpolated_param.RHO_NORM})
      else:
        return da
  else:
    raise ValueError(f"File {path} does not exist.")


class StateHistory:
  """A history of the state of the simulation."""

  def __init__(self, states: tuple[state.ToraxSimState, ...]):
    core_profiles = [state.core_profiles.history_elem() for state in states]
    core_sources = [state.core_sources for state in states]
    transport = [state.core_transport for state in states]
    stack = lambda *ys: jnp.stack(ys)
    self.core_profiles: state.CoreProfiles = jax.tree_util.tree_map(
        stack, *core_profiles
    )
    self.core_sources: source_profiles.SourceProfiles = jax.tree_util.tree_map(
        stack, *core_sources
    )
    self.core_transport: state.CoreTransport = jax.tree_util.tree_map(
        stack, *transport
    )
    self.times = jnp.array([state.t for state in states])
    chex.assert_rank(self.times, 1)

  def _pack_into_data_array(
      self,
      name: str,
      data: jax.Array,
      geo: geometry.Geometry,
  ) -> xr.DataArray | None:
    """Packs the data into an xr.DataArray."""
    is_face_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(geo.rho_face_norm),
    )
    is_cell_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(geo.rho_norm),
    )
    is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times),)

    match data:
      case data if is_face_var(data):
        dims = [TIME, RHO_FACE]
      case data if is_cell_var(data):
        dims = [TIME, RHO_CELL]
      case data if is_scalar(data):
        dims = [TIME]
      case _:
        logging.warning(
            "Unsupported data shape for %s: %s. Skipping persisting.",
            name,
            data.shape,
        )
        return None
    return xr.DataArray(data, dims=dims, name=name)

  def _get_core_profiles(
      self, geo: geometry.Geometry,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core profiles to a dict."""
    xr_dict = {}

    xr_dict[TEMP_EL] = self.core_profiles.temp_el.value
    xr_dict[TEMP_EL_RIGHT_BC] = (
        self.core_profiles.temp_el.right_face_constraint
    )
    xr_dict[TEMP_ION] = self.core_profiles.temp_ion.value
    xr_dict[TEMP_ION_RIGHT_BC] = (
        self.core_profiles.temp_ion.right_face_constraint
    )
    xr_dict[PSI] = self.core_profiles.psi.value
    xr_dict[PSI_RIGHT_GRAD_BC] = (
        self.core_profiles.psi.right_face_grad_constraint
    )
    xr_dict[PSIDOT] = self.core_profiles.psidot.value
    xr_dict[NE] = self.core_profiles.ne.value
    xr_dict[NE_RIGHT_BC] = self.core_profiles.ne.right_face_constraint
    xr_dict[NI] = self.core_profiles.ni.value
    xr_dict[NI_RIGHT_BC] = self.core_profiles.ni.right_face_constraint

    # Currents.
    xr_dict[JTOT] = self.core_profiles.currents.jtot
    xr_dict[JTOT_FACE] = self.core_profiles.currents.jtot_face
    xr_dict[JOHM] = self.core_profiles.currents.johm
    xr_dict[JOHM_FACE] = self.core_profiles.currents.johm_face
    xr_dict[JEXT] = self.core_profiles.currents.jext
    xr_dict[JEXT_FACE] = self.core_profiles.currents.jext_face

    xr_dict[J_BOOTSTRAP] = self.core_profiles.currents.j_bootstrap
    xr_dict[J_BOOTSTRAP_FACE] = self.core_profiles.currents.j_bootstrap_face
    xr_dict[I_BOOTSTRAP] = self.core_profiles.currents.I_bootstrap
    xr_dict[SIGMA] = self.core_profiles.currents.sigma

    xr_dict[Q_FACE] = self.core_profiles.q_face
    xr_dict[S_FACE] = self.core_profiles.s_face
    xr_dict[NREF] = self.core_profiles.nref

    xr_dict = {
        k: self._pack_into_data_array(k, name, geo)
        for k, name in xr_dict.items()
    }

    return xr_dict

  def _save_core_transport(
      self,
      geo: geometry.Geometry,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core transport to a dict."""
    xr_dict = {}

    xr_dict[CHI_FACE_ION] = self.core_transport.chi_face_ion
    xr_dict[CHI_FACE_EL] = self.core_transport.chi_face_el
    xr_dict[D_FACE_EL] = self.core_transport.d_face_el
    xr_dict[V_FACE_EL] = self.core_transport.v_face_el

    xr_dict = {
        k: self._pack_into_data_array(k, name, geo)
        for k, name in xr_dict.items()
    }

    return xr_dict

  def _save_core_sources(
      self,
      geo: geometry.Geometry,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core sources to a dict."""
    xr_dict = {}
    for profile in self.core_sources.profiles:
      xr_dict[profile] = self.core_sources.profiles[profile]

    xr_dict = {
        k: self._pack_into_data_array(k, name, geo)
        for k, name in xr_dict.items()
    }

    return xr_dict

  def simulation_output_to_xr(
      self,
      geo: geometry.Geometry,
  ) -> xr.Dataset:
    """Build an xr.Dataset of the simulation output.

    Args:
      geo: The geometry of the simulation. This is used to retrieve the TORAX
        mesh grid values.

    Returns:
      An xr.Dataset of the simulation output. The dataset contains the following
      coordinates:
        - time: The time of the simulation.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - rho_face: The toroidal coordinate on the face grid.
        - rho_cell: The toroidal coordinate on the cell grid.
      The dataset contains data variables for quantities in the CoreProfiles,
      CoreTransport, and CoreSources.
    """
    # TODO(b/338033916). Extend outputs with:
    # Post-processed integrals, more geo outputs.
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    # Get coordinate variables for dimensions ("time", "rho_face", "rho_cell")
    time = xr.DataArray(self.times, dims=[TIME], name=TIME)
    rho_face_norm = xr.DataArray(
        geo.rho_face_norm, dims=[RHO_FACE], name=RHO_FACE_NORM
    )
    rho_cell_norm = xr.DataArray(
        geo.rho_norm, dims=[RHO_CELL], name=RHO_CELL_NORM
    )
    rho_face = xr.DataArray(geo.rho_face, dims=[RHO_FACE], name=RHO_FACE)
    rho_cell = xr.DataArray(geo.rho, dims=[RHO_CELL], name=RHO_CELL)

    # Initialize dict with desired geometry and reference variables
    xr_dict = {
        VPR: xr.DataArray(geo.vpr, dims=[RHO_CELL], name=VPR),
        SPR: xr.DataArray(geo.spr_cell, dims=[RHO_CELL], name=SPR),
        VPR_FACE: xr.DataArray(geo.vpr_face, dims=[RHO_FACE], name=VPR_FACE),
        SPR_FACE: xr.DataArray(geo.spr_face, dims=[RHO_FACE], name=SPR_FACE),
    }

    xr_dict.update(self._get_core_profiles(geo,))
    xr_dict.update(self._save_core_transport(geo,))
    xr_dict.update(self._save_core_sources(geo,))

    ds = xr.Dataset(
        xr_dict,
        coords={
            TIME: time,
            RHO_FACE_NORM: rho_face_norm,
            RHO_CELL_NORM: rho_cell_norm,
            RHO_FACE: rho_face,
            RHO_CELL: rho_cell,
        },
    )
    return ds
