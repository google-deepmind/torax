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
import inspect

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import state
from torax.config import runtime_params
from torax.geometry import geometry as geometry_lib
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
CORE_PROFILES = "core_profiles"
TEMP_EL = "temp_el"
TEMP_EL_RIGHT_BC = "temp_el_right_bc"
TEMP_ION = "temp_ion"
TEMP_ION_RIGHT_BC = "temp_ion_right_bc"
PSI = "psi"
PSIDOT = "psidot"
PSI_RIGHT_GRAD_BC = "psi_right_grad_bc"
PSI_RIGHT_BC = "psi_right_bc"
NE = "ne"
NE_RIGHT_BC = "ne_right_bc"
NI = "ni"
NI_RIGHT_BC = "ni_right_bc"
JTOT = "jtot"
JTOT_FACE = "jtot_face"
JOHM = "johm"
EXTERNAL_CURRENT = "external_current_source"
J_BOOTSTRAP = "j_bootstrap"
J_BOOTSTRAP_FACE = "j_bootstrap_face"
I_BOOTSTRAP = "I_bootstrap"
SIGMA = "sigma"
Q_FACE = "q_face"
S_FACE = "s_face"
NREF = "nref"
ZIMP = "Zimp"
NIMP = "nimp"
IP_PROFILE_FACE = "Ip_profile_face"
VLOOP_LCFS = "vloop_lcfs"

# Core transport.
CORE_TRANSPORT = "core_transport"
CHI_FACE_ION = "chi_face_ion"
CHI_FACE_EL = "chi_face_el"
D_FACE_EL = "d_face_el"
V_FACE_EL = "v_face_el"

# Geometry.
GEOMETRY = "geometry"

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Post processed outputs
POST_PROCESSED_OUTPUTS = "post_processed_outputs"
Q_FUSION = "Q_fusion"

# Simulation error state.
SIM_ERROR = "sim_error"

# Sources.
CORE_SOURCES = "core_sources"

# Excluded coordinates from geometry since they are at the top DataTree level.
# Exclude q_correction_factor as it is not an interesting quantity to save.
# TODO(b/338033916): consolidate on either rho or rho_cell naming for cell grid
EXCLUDED_GEOMETRY_NAMES = frozenset({
    RHO_CELL,
    RHO_FACE,
    RHO_CELL_NORM,
    RHO_FACE_NORM,
    "rho",
    "rho_norm",
    "q_correction_factor",
})


def safe_load_dataset(filepath: str) -> xr.DataTree:
  with open(filepath, "rb") as f:
    with xr.open_datatree(f) as dt_open:
      data_tree = dt_open.compute()
  return data_tree


def load_state_file(
    filepath: str,
) -> xr.DataTree:
  """Loads a state file from a filepath."""
  if os.path.exists(filepath):
    data_tree = safe_load_dataset(filepath)
    logging.info("Loading state file %s", filepath)
    return data_tree
  else:
    raise ValueError(f"File {filepath} does not exist.")


def concat_datatrees(
    tree1: xr.DataTree,
    tree2: xr.DataTree,
) -> xr.DataTree:
  """Concats two xr.DataTrees along the time dimension.

  For any duplicate time steps, the values from the first dataset are kept.

  Args:
    tree1: The first xr.DataTree to concatenate.
    tree2: The second xr.DataTree to concatenate.

  Returns:
    A xr.DataTree containing the concatenation of the two datasets.
  """

  def _concat_datasets(
      previous_ds: xr.Dataset,
      ds: xr.Dataset,
  ) -> xr.Dataset:
    """Concats two xr.Datasets."""
    # Do a minimal concat to avoid concatting any non time indexed vars.
    ds = xr.concat([previous_ds, ds], dim=TIME, data_vars="minimal")
    # Drop any duplicate time steps. Using "first" imposes
    # keeping the restart state from the earlier dataset. In the case of TORAX
    # restarts this contains more complete information e.g. transport and post
    # processed outputs.
    ds = ds.drop_duplicates(dim=TIME, keep="first")
    return ds

  return xr.map_over_datasets(_concat_datasets, tree1, tree2)


def stitch_state_files(
    file_restart: runtime_params.FileRestart, datatree: xr.DataTree
) -> xr.DataTree:
  """Stitch a datatree to the end of a previous state file.

  Args:
    file_restart: Contains information on a file this sim was restarted from.
    datatree: The xr.DataTree to stitch to the end of the previous state file.

  Returns:
    A xr.DataTree containing the stitched dataset.
  """
  previous_datatree = load_state_file(file_restart.filename)
  # Reduce previous_ds to all times before the first time step in this
  # sim output. We use ds.time[0] instead of file_restart.time because
  # we are uncertain if file_restart.time is the exact time of the
  # first time step in this sim output (it takes the nearest time).
  previous_datatree = previous_datatree.sel(time=slice(None, datatree.time[0]))
  return concat_datatrees(previous_datatree, datatree)


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
    geometries = [state.geometry for state in sim_outputs.sim_history]
    self.geometry = geometry_lib.stack_geometries(geometries)
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
    # The rho grid does not change in time so we can just take the first one.
    self.rho_norm = sim_outputs.sim_history[0].geometry.rho_norm
    self.rho_face_norm = sim_outputs.sim_history[0].geometry.rho_face_norm
    self.rho_face = sim_outputs.sim_history[0].geometry.rho_face
    self.rho = sim_outputs.sim_history[0].geometry.rho
    chex.assert_rank(self.times, 1)
    self.sim_error = sim_outputs.sim_error
    self.source_models = source_models

  def _pack_into_data_array(
      self,
      name: str,
      data: jax.Array | None,
  ) -> xr.DataArray | None:
    """Packs the data into an xr.DataArray."""
    if data is None:
      return None

    is_face_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(self.rho_face_norm),
    )
    is_cell_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(self.rho_norm),
    )
    is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times),)
    is_constant = lambda x: x.ndim == 0

    match data:
      case data if is_face_var(data):
        dims = [TIME, RHO_FACE]
      case data if is_cell_var(data):
        dims = [TIME, RHO_CELL]
      case data if is_scalar(data):
        dims = [TIME]
      case data if is_constant(data):
        dims = []
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
    xr_dict[PSI_RIGHT_BC] = self.core_profiles.psi.right_face_constraint
    xr_dict[PSIDOT] = self.core_profiles.psidot.value
    xr_dict[NE] = self.core_profiles.ne.value
    xr_dict[NE_RIGHT_BC] = self.core_profiles.ne.right_face_constraint
    xr_dict[NI] = self.core_profiles.ni.value
    xr_dict[NI_RIGHT_BC] = self.core_profiles.ni.right_face_constraint
    xr_dict[ZIMP] = self.core_profiles.Zimp
    xr_dict[NIMP] = self.core_profiles.nimp.value

    # Currents.
    xr_dict[JTOT] = self.core_profiles.currents.jtot
    xr_dict[JTOT_FACE] = self.core_profiles.currents.jtot_face
    xr_dict[JOHM] = self.core_profiles.currents.johm
    xr_dict[EXTERNAL_CURRENT] = (
        self.core_profiles.currents.external_current_source
    )

    xr_dict[J_BOOTSTRAP] = self.core_profiles.currents.j_bootstrap
    xr_dict[J_BOOTSTRAP_FACE] = self.core_profiles.currents.j_bootstrap_face
    xr_dict[IP_PROFILE_FACE] = self.core_profiles.currents.Ip_profile_face
    xr_dict[I_BOOTSTRAP] = self.core_profiles.currents.I_bootstrap
    xr_dict[SIGMA] = self.core_profiles.currents.sigma

    xr_dict[Q_FACE] = self.core_profiles.q_face
    xr_dict[S_FACE] = self.core_profiles.s_face
    xr_dict[NREF] = self.core_profiles.nref

    xr_dict[VLOOP_LCFS] = self.core_profiles.vloop_lcfs

    xr_dict = {
        name: self._pack_into_data_array(
            name,
            data,
        )
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_core_transport(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core transport to a dict."""
    xr_dict = {}

    xr_dict[CHI_FACE_ION] = self.core_transport.chi_face_ion
    xr_dict[CHI_FACE_EL] = self.core_transport.chi_face_el
    xr_dict[D_FACE_EL] = self.core_transport.d_face_el
    xr_dict[V_FACE_EL] = self.core_transport.v_face_el

    xr_dict = {
        name: self._pack_into_data_array(
            name,
            data,
        )
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_core_sources(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core sources to a dict."""
    xr_dict = {}

    xr_dict[self.source_models.qei_source_name] = (
        self.core_sources.qei.qei_coef
        * (self.core_profiles.temp_el.value - self.core_profiles.temp_ion.value)
    )

    # Add source profiles with suffixes indicating which profile they affect.
    for profile in self.core_sources.temp_ion:
      xr_dict[f"{profile}_ion"] = self.core_sources.temp_ion[profile]
    for profile in self.core_sources.temp_el:
      xr_dict[f"{profile}_el"] = self.core_sources.temp_el[profile]
    for profile in self.core_sources.psi:
      xr_dict[f"{profile}_j"] = self.core_sources.psi[profile]
    for profile in self.core_sources.ne:
      xr_dict[f"{profile}_ne"] = self.core_sources.ne[profile]

    xr_dict = {
        name: self._pack_into_data_array(name, data)
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_post_processed_outputs(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the post processed outputs to a dict."""
    xr_dict = {}
    for field_name, data in dataclasses.asdict(
        self.post_processed_outputs
    ).items():
      xr_dict[field_name] = self._pack_into_data_array(field_name, data)
    return xr_dict

  def _save_geometry(
      self,
  ) -> dict[str, xr.DataArray]:
    """Save geometry to a dict. We skip over hires and non-array quantities."""
    xr_dict = {}

    # Get the variables from dataclass fields.
    for field_name, data in dataclasses.asdict(self.geometry).items():
      if (
          "hires" in field_name
          or field_name == "geometry_type"
          or field_name == "Ip_from_parameters"
          or not isinstance(data, jax.Array)
      ):
        continue
      data_array = self._pack_into_data_array(
          field_name,
          data,
      )
      if data_array is not None:
        xr_dict[field_name] = data_array

    # Get variables from property methods

    for name, value in inspect.getmembers(type(self.geometry)):
      if name in EXCLUDED_GEOMETRY_NAMES:
        continue
      if isinstance(value, property):
        property_data = value.fget(self.geometry)
        data_array = self._pack_into_data_array(name, property_data)
        if data_array is not None:
          xr_dict[name] = data_array

    return xr_dict

  def simulation_output_to_xr(
      self,
      file_restart: runtime_params.FileRestart | None = None,
  ) -> xr.DataTree:
    """Build an xr.DataTree of the simulation output.

    Args:
      file_restart: If provided, contains information on a file this sim was
        restarted from, this is useful in case we want to stitch that to the
        beggining of this sim output.

    Returns:
      A xr.DataTree containing a single top level xr.Dataset and four child
      datasets. The top level dataset contains the following variables:
        - time: The time of the simulation.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - rho_face: The toroidal coordinate on the face grid.
        - rho_cell: The toroidal coordinate on the cell grid.
        - sim_error: The simulation error state.
      The child datasets contain the following variables:
        - core_profiles: Contains data variables for quantities in the
          CoreProfiles.
        - core_transport: Contains data variables for quantities in the
          CoreTransport.
        - core_sources: Contains data variables for quantities in the
          CoreSources.
        - post_processed_outputs: Contains data variables for quantities in the
          PostProcessedOutputs.
        - geometry: Contains data variables for quantities in the Geometry.
    """
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    # Get coordinate variables for dimensions ("time", "rho_face", "rho_cell")
    time = xr.DataArray(self.times, dims=[TIME], name=TIME)
    rho_face_norm = xr.DataArray(
        self.rho_face_norm, dims=[RHO_FACE], name=RHO_FACE_NORM
    )
    rho_cell_norm = xr.DataArray(
        self.rho_norm, dims=[RHO_CELL], name=RHO_CELL_NORM
    )
    rho_face = xr.DataArray(self.rho_face, dims=[RHO_FACE], name=RHO_FACE)
    rho_cell = xr.DataArray(self.rho, dims=[RHO_CELL], name=RHO_CELL)

    coords = {
        TIME: time,
        RHO_FACE_NORM: rho_face_norm,
        RHO_CELL_NORM: rho_cell_norm,
        RHO_FACE: rho_face,
        RHO_CELL: rho_cell,
    }

    # Update dict with flattened StateHistory dataclass containers
    core_profiles_ds = xr.Dataset(self._get_core_profiles(), coords=coords)
    core_transport_ds = xr.Dataset(self._save_core_transport(), coords=coords)
    core_sources_ds = xr.Dataset(
        self._save_core_sources(),
        coords=coords,
    )
    post_processed_outputs_ds = xr.Dataset(
        self._save_post_processed_outputs(), coords=coords
    )
    geometry_ds = xr.Dataset(self._save_geometry(), coords=coords)
    top_level_xr_dict = {SIM_ERROR: self.sim_error.value}
    data_tree = xr.DataTree(
        children={
            CORE_PROFILES: xr.DataTree(dataset=core_profiles_ds),
            CORE_TRANSPORT: xr.DataTree(dataset=core_transport_ds),
            CORE_SOURCES: xr.DataTree(dataset=core_sources_ds),
            POST_PROCESSED_OUTPUTS: xr.DataTree(
                dataset=post_processed_outputs_ds
            ),
            GEOMETRY: xr.DataTree(dataset=geometry_ds),
        },
        dataset=xr.Dataset(top_level_xr_dict, coords=coords),
    )

    if file_restart is not None and file_restart.stitch:
      data_tree = stitch_state_files(file_restart, data_tree)

    return data_tree
