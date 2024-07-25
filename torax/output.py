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
from typing import Any

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.sources import source_profiles
import xarray as xr

import os


Filepath = tuple[str, float]

TEMP_EL_VALUE = "temp_el"
TEMP_EL_RIGHT_BC = "temp_el_right_bc"
TEMP_ION_VALUE = "temp_ion"
TEMP_ION_RIGHT_BC = "temp_ion_right_bc"
PSI_VALUE = "psi"
PSI_RIGHT_GRAD_BC = "psi_right_grad_bc"
PSIDOT_VALUE = "psidot"
NE_VALUE = "ne"
NE_RIGHT_BC = "ne_right_bc"
NI_VALUE = "ni"
NI_RIGHT_BC = "ni_right_bc"
JTOT_VALUE = "jtot"
JTOT_FACE_VALUE = "jtot_face"
JOMH_VALUE = "johm"
JOMH_FACE_VALUE = "johm_face"
JEXT_VALUE = "jext"
JEXT_FACE_VALUE = "jext_face"
J_BOOTSTRAP_VALUE = "j_bootstrap"
J_BOOTSTRAP_FACE_VALUE = "j_bootstrap_face"
I_BOOTSTRAP_VALUE = "I_bootstrap"
SIGMA_VALUE = "sigma"
Q_FACE_VALUE = "q_face"
S_FACE_VALUE = "s_face"
NREF_VALUE = "nref"
RHO_NORM = "rho_norm"


def is_torax_xarray_filepath(filepath: Any) -> bool:
  """Returns whether the filepath is a valid Torax xarray filepath."""
  if isinstance(filepath, tuple) and len(filepath) == 2:
    path, t = filepath
    if os.path.exists(path) and isinstance(t, float):
      return True
  return False


def load_state_file(filepath: Filepath, data_var: str) -> xr.DataArray:
  """Loads a state file from a filepath."""
  path, t = filepath
  if os.path.exists(path):
    with open(path, "rb") as f:
      logging.info("Loading %s from state file %s, time %s", data_var, path, t)
      da = xr.load_dataset(f).sel(time=slice(t, None)).data_vars[data_var]
      if data_var.endswith("bc"):
        return da
      return da.rename({"r_cell_norm": RHO_NORM})
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
        len(geo.r_face),
    )
    is_cell_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(geo.r),
    )
    is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times),)

    match data:
      case data if is_face_var(data):
        dims = ["time", "rho_face"]
      case data if is_cell_var(data):
        dims = ["time", "rho_cell"]
      case data if is_scalar(data):
        dims = ["time"]
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

    xr_dict[TEMP_EL_VALUE] = self.core_profiles.temp_el.value
    xr_dict[TEMP_EL_RIGHT_BC] = (
        self.core_profiles.temp_el.right_face_constraint
    )
    xr_dict[TEMP_ION_VALUE] = self.core_profiles.temp_ion.value
    xr_dict[TEMP_ION_RIGHT_BC] = (
        self.core_profiles.temp_ion.right_face_constraint
    )
    xr_dict[PSI_VALUE] = self.core_profiles.psi.value
    xr_dict[PSI_RIGHT_GRAD_BC] = (
        self.core_profiles.psi.right_face_grad_constraint
    )
    xr_dict[PSIDOT_VALUE] = self.core_profiles.psidot.value
    xr_dict[NE_VALUE] = self.core_profiles.ne.value
    xr_dict[NE_RIGHT_BC] = self.core_profiles.ne.right_face_constraint
    xr_dict[NI_VALUE] = self.core_profiles.ni.value
    xr_dict[NI_RIGHT_BC] = self.core_profiles.ni.right_face_constraint

    # Currents.
    xr_dict[JTOT_VALUE] = self.core_profiles.currents.jtot
    xr_dict[JTOT_FACE_VALUE] = self.core_profiles.currents.jtot_face
    xr_dict[JOMH_VALUE] = self.core_profiles.currents.johm
    xr_dict[JOMH_FACE_VALUE] = self.core_profiles.currents.johm_face
    xr_dict[JEXT_VALUE] = self.core_profiles.currents.jext
    xr_dict[JEXT_FACE_VALUE] = self.core_profiles.currents.jext_face

    xr_dict[J_BOOTSTRAP_VALUE] = self.core_profiles.currents.j_bootstrap
    xr_dict[J_BOOTSTRAP_FACE_VALUE] = (
        self.core_profiles.currents.j_bootstrap_face
    )
    xr_dict[I_BOOTSTRAP_VALUE] = self.core_profiles.currents.I_bootstrap
    xr_dict[SIGMA_VALUE] = self.core_profiles.currents.sigma

    xr_dict[Q_FACE_VALUE] = self.core_profiles.q_face
    xr_dict[S_FACE_VALUE] = self.core_profiles.s_face
    xr_dict[NREF_VALUE] = self.core_profiles.nref

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

    xr_dict["chi_face_ion"] = self.core_transport.chi_face_ion
    xr_dict["chi_face_el"] = self.core_transport.chi_face_el
    xr_dict["d_face_el"] = self.core_transport.d_face_el
    xr_dict["v_face_el"] = self.core_transport.v_face_el

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
    """Build an xr.Dataset of the simulation output."""

    # TODO(b/338033916). Extend outputs with:
    # Post-processed integrals, more geo outputs.
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    # Get coordinate variables for dimensions ("time", "rho_face", "rho_cell")
    time = xr.DataArray(self.times, dims=["time"], name="time")
    r_face_norm = xr.DataArray(
        geo.r_face_norm, dims=["rho_face"], name="r_face_norm"
    )
    r_cell_norm = xr.DataArray(
        geo.r_norm, dims=["rho_cell"], name="r_cell_norm"
    )
    r_face = xr.DataArray(geo.r_face, dims=["rho_face"], name="r_face")
    r_cell = xr.DataArray(geo.r, dims=["rho_cell"], name="r_cell")

    # Initialize dict with desired geometry and reference variables
    xr_dict = {
        "vpr": xr.DataArray(geo.vpr, dims=["rho_cell"], name="vpr"),
        "spr": xr.DataArray(geo.spr_cell, dims=["rho_cell"], name="spr"),
        "vpr_face": xr.DataArray(
            geo.vpr_face, dims=["rho_face"], name="vpr_face"
        ),
        "spr_face": xr.DataArray(
            geo.spr_face, dims=["rho_face"], name="spr_face"
        ),
    }

    xr_dict.update(self._get_core_profiles(geo,))
    xr_dict.update(self._save_core_transport(geo,))
    xr_dict.update(self._save_core_sources(geo,))

    ds = xr.Dataset(
        xr_dict,
        coords={
            "time": time,
            "r_face_norm": r_face_norm,
            "r_cell_norm": r_cell_norm,
            "r_face": r_face,
            "r_cell": r_cell,
        },
    )
    return ds
