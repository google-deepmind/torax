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

from collections.abc import Sequence
import dataclasses
import itertools

import os

from absl import logging
import chex
import jax
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.edge import base as edge_base
from torax._src.geometry import geometry as geometry_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.output_tools import post_processing
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import file_restart as file_restart_config
from torax._src.torax_pydantic import model_config
import xarray as xr

# Internal import.


def load_state_file(filepath: str) -> xr.DataTree:
  """Loads a state file from a filepath."""
  if os.path.exists(filepath):
    # necessary to use open here to work reliably in colab.
    with open(filepath, "rb") as f:
      dt_open = xr.open_datatree(f)
      data_tree = dt_open.compute()
    logging.info("Loaded state file %s", filepath)
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
    ds = xr.concat([previous_ds, ds], dim=output_keys.TIME, data_vars="minimal")
    # Drop any duplicate time steps. Using "first" imposes
    # keeping the restart state from the earlier dataset. In the case of TORAX
    # restarts this contains more complete information e.g. transport and post
    # processed outputs.
    ds = ds.drop_duplicates(dim=output_keys.TIME, keep="first")
    return ds

  return xr.map_over_datasets(_concat_datasets, tree1, tree2)


def stitch_state_files(
    file_restart: file_restart_config.FileRestartConfig,
    datatree: xr.DataTree,
) -> xr.DataTree:
  """Stitch a datatree to the end of a previous state file.

  Args:
    file_restart: Contains information on a file this sim was restarted from.
    datatree: The xr.DataTree to stitch to the end of the previous state file.

  Returns:
    A xr.DataTree containing the stitched dataset.
  """
  previous_datatree = load_state_file(file_restart.filename)  # pyrefly: ignore[bad-argument-type]
  np.testing.assert_array_equal(
      previous_datatree.coords[output_keys.RHO_CELL_NORM].as_numpy(),
      datatree.coords[output_keys.RHO_CELL_NORM].as_numpy(),
      err_msg=(
          "The rho_cell_norm coordinates of the previous state file and the"
          " current state file must be the same."
      ),
  )
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
      state_history: list[sim_state.SimState],
      post_processed_outputs_history: tuple[
          post_processing.PostProcessedOutputs, ...
      ],
      sim_error: state.SimError,
      torax_config: model_config.ToraxConfig,
  ):
    self._sim_error = sim_error
    self._torax_config = torax_config
    self._post_processed_outputs = post_processed_outputs_history
    self._solver_numeric_outputs = [
        state.solver_numeric_outputs for state in state_history
    ]
    self._core_profiles = [state.core_profiles for state in state_history]
    if (
        not torax_config.restart
        and not torax_config.profile_conditions.use_v_loop_lcfs_boundary_condition
        and len(state_history) >= 2
    ):
      # For the Ip BC case, set v_loop_lcfs[0] to the same value as
      # v_loop_lcfs[1] due the v_loop_lcfs timeseries being
      # underconstrained.
      self._core_profiles[0] = dataclasses.replace(
          self._core_profiles[0],
          v_loop_lcfs=self._core_profiles[1].v_loop_lcfs,
      )
    self._core_sources = [state.core_sources for state in state_history]
    self._edge_outputs = [state.edge_outputs for state in state_history]
    self._transport = [state.core_transport for state in state_history]
    self._geometries = [state.geometry for state in state_history]
    self._model_states = [state.model_state for state in state_history]
    self._stacked_geometry = geometry_lib.stack_geometries(self.geometries)
    stack = lambda *ys: np.stack(ys)
    self._stacked_core_profiles: state.CoreProfiles = jax.tree.map(
        stack, *self._core_profiles
    )
    self._stacked_core_sources: source_profiles_lib.SourceProfiles = (
        jax.tree.map(stack, *self._core_sources)
    )
    self._stacked_core_transport: state.CoreTransport = jax.tree.map(
        stack, *self._transport
    )
    self._stacked_post_processed_outputs: (
        post_processing.PostProcessedOutputs
    ) = jax.tree.map(stack, *post_processed_outputs_history)
    self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
        jax.tree.map(stack, *self._solver_numeric_outputs)
    )
    # If self._edge_outputs is a list of Nones, stacked_edge_outputs will just
    # be None, since jax.tree.map treats None as a node with no leaves.
    self._stacked_edge_outputs: edge_base.EdgeModelOutputs = jax.tree.map(
        stack, *self._edge_outputs
    )

    self._times = np.array([state.t for state in state_history])
    chex.assert_rank(self.times, 1)
    # The rho grid does not change in time so we can just take the first one.
    self._rho_cell_norm = state_history[0].geometry.rho_norm
    self._rho_face_norm = state_history[0].geometry.rho_face_norm
    self._rho_norm = np.concatenate([[0.0], self.rho_cell_norm, [1.0]])

    self._output_grid_context = output_grid_context.OutputGridContext(
        times=self.times,
        rho_face_norm=self.rho_face_norm,
        rho_cell_norm=self.rho_cell_norm,
        rho_cell_plus_boundaries_norm=self.rho_norm,
    )

  @property
  def torax_config(self) -> model_config.ToraxConfig:
    """Returns the ToraxConfig used to run the simulation."""
    return self._torax_config

  @property
  def sim_error(self) -> state.SimError:
    """Returns the simulation error state."""
    return self._sim_error

  @property
  def times(self) -> array_typing.Array:
    """Returns the time of the simulation."""
    return self._times

  @property
  def rho_cell_norm(self) -> array_typing.FloatVectorCell:
    """Returns the normalized toroidal coordinate on the cell grid."""
    return self._rho_cell_norm

  @property
  def rho_face_norm(self) -> array_typing.FloatVectorFace:
    """Returns the normalized toroidal coordinate on the face grid."""
    return self._rho_face_norm

  @property
  def rho_norm(self) -> array_typing.FloatVectorCellPlusBoundaries:
    """Returns the rho on the cell grid with the left and right face boundaries."""
    return self._rho_norm

  @property
  def geometries(self) -> Sequence[geometry_lib.Geometry]:
    """Returns the geometries of the simulation."""
    return self._geometries

  @property
  def core_profiles(self) -> Sequence[state.CoreProfiles]:
    """Returns the core profiles."""
    return self._core_profiles

  @property
  def source_profiles(self) -> Sequence[source_profiles_lib.SourceProfiles]:
    """Returns the source profiles for the simulation."""
    return self._core_sources

  @property
  def core_transport(self) -> Sequence[state.CoreTransport]:
    """Returns the core transport for the simulation."""
    return self._transport

  @property
  def solver_numeric_outputs(self) -> Sequence[state.SolverNumericOutputs]:
    """Returns the solver numeric outputs."""
    return self._solver_numeric_outputs

  @property
  def post_processed_outputs(
      self,
  ) -> Sequence[post_processing.PostProcessedOutputs]:
    """Returns the post processed outputs for the simulation."""
    return self._post_processed_outputs

  def simulation_output_to_xr(self) -> xr.DataTree:
    """Build an xr.DataTree of the simulation output.

    Returns:
      A xr.DataTree containing a single top level xr.Dataset and four child
      datasets. The top level dataset contains the following variables:
        - time: The time of the simulation.
        - rho_norm: The normalized toroidal coordinate on the cell grid with the
            left and right face boundaries added.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - config: The ToraxConfig used to run the simulation serialized to JSON.
      The child datasets contain the following variables:
        - numerics: Contains data variables for numeric quantities to do with
            the simulation.
        - profiles: Contains data variables for 1D profiles.
        - scalars: Contains data variables for scalars.
        - edge: Contains data variables for the edge model, if one is active.
    """

    time = xr.DataArray(
        self.times,
        dims=[output_keys.TIME],
        name=output_keys.TIME,
        attrs=output_keys.get_units(output_keys.TIME),
    )
    rho_face_norm = xr.DataArray(
        self.rho_face_norm,
        dims=[output_keys.RHO_FACE_NORM],
        name=output_keys.RHO_FACE_NORM,
        attrs=output_keys.get_units(output_keys.RHO_FACE_NORM),
    )
    rho_cell_norm = xr.DataArray(
        self.rho_cell_norm,
        dims=[output_keys.RHO_CELL_NORM],
        name=output_keys.RHO_CELL_NORM,
        attrs=output_keys.get_units(output_keys.RHO_CELL_NORM),
    )
    rho_norm = xr.DataArray(
        self.rho_norm,
        dims=[output_keys.RHO_NORM],
        name=output_keys.RHO_NORM,
        attrs=output_keys.get_units(output_keys.RHO_NORM),
    )

    coords = {
        output_keys.TIME: time,
        output_keys.RHO_FACE_NORM: rho_face_norm,
        output_keys.RHO_CELL_NORM: rho_cell_norm,
        output_keys.RHO_NORM: rho_norm,
    }

    # Update dict with flattened StateHistory dataclass containers.
    all_core_data = [
        self._save_core_profiles(),
        self._save_core_transport(),
        self._save_core_sources(),
        self._save_post_processed_outputs(),
        self._save_geometry(),
    ]
    flattened_all_core_data = {}
    for key, value in itertools.chain(*(d.items() for d in all_core_data)):
      if key not in flattened_all_core_data:
        flattened_all_core_data[key] = value
      else:
        raise ValueError(f"Duplicate key: {key}")

    # Determine simulation status based on error state.
    sim_status = (
        state.SimStatus.COMPLETED
        if self.sim_error is state.SimError.NO_ERROR
        else state.SimStatus.ERROR
    )

    numerics_vars: dict[str, output_grid_context.OutputVar] = {
        output_keys.SIM_STATUS: (
            (),
            np.asarray(sim_status.value),
            output_keys.get_units(output_keys.SIM_STATUS),
        ),
        output_keys.SIM_ERROR: (
            (),
            np.asarray(self.sim_error.value),
            output_keys.get_units(output_keys.SIM_ERROR),
        ),
    }
    # Populate solver numeric variables from stacked solver outputs.
    numerics_vars.update(
        self._stacked_solver_numeric_outputs.to_output_dict(
            self._output_grid_context
        )
    )
    numerics = self._output_grid_context.build_dataset(numerics_vars)

    spatial_coords = {
        output_keys.RHO_FACE_NORM,
        output_keys.RHO_CELL_NORM,
        output_keys.RHO_NORM,
    }

    profiles_dict = {
        k: v
        for k, v in flattened_all_core_data.items()
        if any(d in spatial_coords for d in v[0])
    }
    profiles = self._output_grid_context.build_dataset(profiles_dict)
    scalars_dict = {
        k: v
        for k, v in flattened_all_core_data.items()
        if not any(d in spatial_coords for d in v[0])
    }
    scalars = self._output_grid_context.build_dataset(scalars_dict)
    children = {
        output_keys.PROFILES: xr.DataTree(dataset=profiles),
        output_keys.NUMERICS: xr.DataTree(dataset=numerics),
        output_keys.SCALARS: xr.DataTree(dataset=scalars),
    }
    if self._stacked_edge_outputs is not None:
      children[output_keys.EDGE] = self._save_edge_outputs()
    turbulent_transport_tree = self._save_turbulent_transport()
    auxiliary_children = {}
    if turbulent_transport_tree.children:
      auxiliary_children[output_keys.TURBULENT_TRANSPORT] = (
          turbulent_transport_tree
      )
    if auxiliary_children:
      children[output_keys.AUXILIARY] = xr.DataTree(
          children=auxiliary_children
      )
    data_tree = xr.DataTree(
        children=children,  # pyrefly: ignore[bad-argument-type]
        dataset=xr.Dataset(
            data_vars=None,
            coords=coords,
            attrs={output_keys.CONFIG: self.torax_config.model_dump_json()},
        ),
    )
    if (
        self.torax_config.restart is not None
        and self.torax_config.restart.stitch
    ):
      data_tree = stitch_state_files(self.torax_config.restart, data_tree)

    return data_tree

  def _save_core_profiles(
      self,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Saves the stacked core profiles to a dictionary."""
    return self._stacked_core_profiles.to_output_dict(
        self._output_grid_context
    )

  def _save_core_transport(
      self,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Saves the core transport to a dict."""
    return self._stacked_core_transport.to_output_dict(
        self._output_grid_context
    )

  def _save_core_sources(
      self,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Saves the core sources to a dict."""
    return self._stacked_core_sources.to_output_dict(
        self._output_grid_context
    )

  def _save_post_processed_outputs(
      self,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Saves the post processed outputs to a dict."""
    return self._stacked_post_processed_outputs.to_output_dict(
        self._output_grid_context
    )

  def _save_geometry(
      self,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Saves the geometry to a dict."""
    return self._stacked_geometry.to_output_dict(self._output_grid_context)

  def _save_edge_outputs(self) -> xr.DataTree:
    """Saves the edge outputs to a DataTree."""
    if self._stacked_edge_outputs is None:
      return xr.DataTree(dataset=xr.Dataset({}))
    return self._stacked_edge_outputs.to_xr_datatree(self._output_grid_context)

  def _save_turbulent_transport(self) -> xr.DataTree:
    """Saves turbulent transport per-model outputs to a DataTree."""
    return self._stacked_core_transport.turbulent.to_xr_datatree(
        self._output_grid_context
    )
