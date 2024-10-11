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
from __future__ import annotations

import dataclasses

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
import xarray as xr

import os


@chex.dataclass(frozen=True)
class ToraxSimOutputs:
  """Output structure returned by `run_simulation()`.

  Contains the error state and the history of the simulation state.
  Can be extended in the future to include more metadata about the simulation.

  Attributes:
    sim_error: simulation error state: NO_ERROR for no error, NAN_DETECTED for
      NaNs found in core profiles.
    sim_history: history of the simulation state.
  """

  # Error state
  sim_error: state.SimError

  # Time-dependent TORAX outputs
  sim_history: tuple[state.ToraxSimState, ...]


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
# TODO(b/338033916): rename when we have a solution for hierarchical outputs.
# Add `core_profiles` prefix here to avoid name clash with core_sources.jext.
CORE_PROFILES_JEXT = "core_profiles_jext"
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
IP = "Ip"

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Post processed outputs
Q_FUSION = "Q_fusion"

# Simulation error state.
SIM_ERROR = "sim_error"


def safe_load_dataset(filepath: str) -> xr.Dataset:
  with open(filepath, "rb") as f:
    with xr.open_dataset(f) as ds_open:
      ds = ds_open.compute()
  return ds


def load_state_file(
    filepath: str,
) -> xr.Dataset:
  """Loads a state file from a filepath."""
  if os.path.exists(filepath):
    ds = safe_load_dataset(filepath)
    logging.info("Loading state file %s", filepath)
    return ds
  else:
    raise ValueError(f"File {filepath} does not exist.")


class StateHistory:
  """A history of the state of the simulation and its error state."""

  def __init__(
      self,
      sim_outputs: ToraxSimOutputs,
      source_models: source_models_lib.SourceModels,
  ):
    core_profiles = [
        state.core_profiles.history_elem() for state in sim_outputs.sim_history
    ]
    core_sources = [state.core_sources for state in sim_outputs.sim_history]
    transport = [state.core_transport for state in sim_outputs.sim_history]
    post_processed_output = [
        state.post_processed_outputs for state in sim_outputs.sim_history
    ]
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
    self.post_processed_outputs: state.PostProcessedOutputs = (
        jax.tree_util.tree_map(stack, *post_processed_output)
    )
    self.times = jnp.array([state.t for state in sim_outputs.sim_history])
    chex.assert_rank(self.times, 1)
    self.sim_error = sim_outputs.sim_error
    self.source_models = source_models

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
      self,
      geo: geometry.Geometry,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core profiles to a dict."""
    xr_dict = {}

    xr_dict[TEMP_EL] = self.core_profiles.temp_el.value
    xr_dict[TEMP_EL_RIGHT_BC] = self.core_profiles.temp_el.right_face_constraint
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
    xr_dict[CORE_PROFILES_JEXT] = self.core_profiles.currents.jext
    xr_dict[JEXT_FACE] = self.core_profiles.currents.jext_face

    xr_dict[J_BOOTSTRAP] = self.core_profiles.currents.j_bootstrap
    xr_dict[J_BOOTSTRAP_FACE] = self.core_profiles.currents.j_bootstrap_face
    xr_dict[IP] = self.core_profiles.currents.Ip
    xr_dict[I_BOOTSTRAP] = self.core_profiles.currents.I_bootstrap
    xr_dict[SIGMA] = self.core_profiles.currents.sigma

    xr_dict[Q_FACE] = self.core_profiles.q_face
    xr_dict[S_FACE] = self.core_profiles.s_face
    xr_dict[NREF] = self.core_profiles.nref

    xr_dict = {
        name: self._pack_into_data_array(name, data, geo)
        for name, data in xr_dict.items()
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
        name: self._pack_into_data_array(name, data, geo)
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_core_sources(
      self,
      geo: geometry.Geometry,
      existing_keys: set[str],
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core sources to a dict."""
    xr_dict = {}

    xr_dict[self.source_models.qei_source_name] = (
        self.core_sources.qei.qei_coef
        * (self.core_profiles.temp_el.value - self.core_profiles.temp_ion.value)
    )

    # Add source profiles
    for profile in self.core_sources.profiles:
      if profile in existing_keys:
        logging.warning(
            "Overlapping key %s between sources and other data structures",
            profile,
        )
      if profile in self.source_models.ion_el_sources:
        xr_dict[f"{profile}_ion"] = self.core_sources.profiles[profile][
            :, 0, ...
        ]
        xr_dict[f"{profile}_el"] = self.core_sources.profiles[profile][
            :, 1, ...
        ]
      else:
        xr_dict[profile] = self.core_sources.profiles[profile]

    xr_dict = {
        name: self._pack_into_data_array(name, data, geo)
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_post_processed_outputs(
      self,
      geo: geometry.Geometry,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the post processed outputs to a dict."""
    xr_dict = {}
    for field_name, data in dataclasses.asdict(
        self.post_processed_outputs
    ).items():
      xr_dict[field_name] = self._pack_into_data_array(field_name, data, geo)

    return xr_dict

  def simulation_output_to_xr(
      self,
      geo: geometry.Geometry,
      file_restart: runtime_params.FileRestart | None = None,
  ) -> xr.Dataset:
    """Build an xr.Dataset of the simulation output.

    Args:
      geo: The geometry of the simulation. This is used to retrieve the TORAX
        mesh grid values.
      file_restart: If provided, contains information on a file this sim was
        restarted from, this is useful in case we want to stitch that to the
        beggining of this sim output.

    Returns:
      An xr.Dataset of the simulation output. The dataset contains the following
      coordinates:
        - time: The time of the simulation.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - rho_face: The toroidal coordinate on the face grid.
        - rho_cell: The toroidal coordinate on the cell grid.
      The dataset contains data variables for quantities in the CoreProfiles,
      CoreTransport, and CoreSources, as well as time and the sim_error state.
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

    # Update dict with flattened StateHistory dataclass containers
    xr_dict.update(self._get_core_profiles(geo))
    xr_dict.update(self._save_core_transport(geo))
    existing_keys = set(xr_dict.keys())
    xr_dict.update(self._save_core_sources(geo, existing_keys))
    xr_dict.update(self._save_post_processed_outputs(geo))

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

    if file_restart is not None and file_restart.stitch:
      previous_ds = load_state_file(
          file_restart.filename,
      )
      # Do a minimal concat to avoid concatting any non time indexed vars.
      ds = xr.concat([previous_ds, ds], dim=TIME, data_vars="minimal")
      ds = ds.drop_duplicates(dim=TIME)

    # Add sim_error as a new variable
    ds[SIM_ERROR] = self.sim_error.value

    return ds
