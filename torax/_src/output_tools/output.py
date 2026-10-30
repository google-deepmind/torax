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
import functools
from typing import Any

import os

from absl import logging
import chex
import jax
import jax.numpy as jnp
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
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import file_restart as file_restart_config
import xarray as xr

# Internal import.


def _unstack_pytree(tree: Any) -> tuple[Any, ...]:
  """Unstacks a PyTree with stacked array leaves along the leading axis."""
  leaves, treedef = jax.tree.flatten(tree)
  if not leaves:
    return ()
  num_steps = leaves[0].shape[0]
  unstacked_leaves = [jnp.unstack(x) for x in leaves]
  return tuple(
      jax.tree.unflatten(treedef, [leaf[i] for leaf in unstacked_leaves])
      for i in range(num_steps)
  )


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
      state_history: sim_state.SimState,
      post_processed_outputs_history: post_processing.PostProcessedOutputs,
      sim_error: state.SimError,
      torax_config: config.ToraxConfig,
  ):
    self._sim_error = sim_error
    self._torax_config = torax_config
    self._times = jnp.asarray(state_history.t)
    chex.assert_rank(self.times, 1)
    core_profiles = state_history.core_profiles
    if (
        not torax_config.restart
        and not torax_config.profile_conditions.use_v_loop_lcfs_boundary_condition
        and len(self._times) >= 2
    ):
      # For the I_p BC case, set v_loop_lcfs[0] to the same value as
      # v_loop_lcfs[1] due the v_loop_lcfs timeseries being
      # underconstrained.
      v_loop_lcfs = jnp.asarray(core_profiles.v_loop_lcfs)
      core_profiles = dataclasses.replace(
          core_profiles,
          v_loop_lcfs=v_loop_lcfs.at[0].set(v_loop_lcfs[1]),
      )
    self._core_profiles: state.CoreProfiles = core_profiles
    self._core_sources: source_profiles_lib.SourceProfiles = (
        state_history.core_sources
    )
    self._stacked_core_transport: state.CoreTransport = (
        state_history.core_transport
    )
    self._stacked_post_processed_outputs: (
        post_processing.PostProcessedOutputs
    ) = post_processed_outputs_history
    self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
        state_history.solver_numeric_outputs
    )
    self._stacked_edge_outputs: edge_base.EdgeModelOutputs | None = (
        state_history.edge_outputs
    )
    self._geometry: geometry_lib.Geometry = state_history.geometry

    # The rho grid does not change in time so we can just take the first one.
    self._rho_cell_norm = state_history.geometry.rho_norm[0]
    self._rho_face_norm = state_history.geometry.rho_face_norm[0]
    self._rho_norm = jnp.concatenate([
        jnp.zeros(1, dtype=self.rho_cell_norm.dtype),
        self.rho_cell_norm,
        jnp.ones(1, dtype=self.rho_cell_norm.dtype),
    ])

  @property
  def _output_grid_context(self) -> output_grid_context.OutputGridContext:
    return output_grid_context.OutputGridContext(
        times=self.times,
        rho_face_norm=self.rho_face_norm,
        rho_cell_norm=self.rho_cell_norm,
        rho_cell_plus_boundaries_norm=self.rho_norm,
    )

  @property
  def torax_config(self) -> config.ToraxConfig:
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

  @functools.cached_property
  def geometries(self) -> Sequence[geometry_lib.Geometry]:
    """Returns the geometries of the simulation."""
    return _unstack_pytree(self._geometry)

  @functools.cached_property
  def core_profiles(self) -> Sequence[state.CoreProfiles]:
    """Returns the core profiles."""
    return _unstack_pytree(self._core_profiles)

  @functools.cached_property
  def source_profiles(self) -> Sequence[source_profiles_lib.SourceProfiles]:
    """Returns the source profiles for the simulation."""
    return _unstack_pytree(self._core_sources)

  @functools.cached_property
  def core_transport(self) -> Sequence[state.CoreTransport]:
    """Returns the core transport for the simulation."""
    return _unstack_pytree(self._stacked_core_transport)

  @functools.cached_property
  def solver_numeric_outputs(self) -> Sequence[state.SolverNumericOutputs]:
    """Returns the solver numeric outputs."""
    return _unstack_pytree(self._stacked_solver_numeric_outputs)

  @functools.cached_property
  def post_processed_outputs(
      self,
  ) -> Sequence[post_processing.PostProcessedOutputs]:
    """Returns the post processed outputs for the simulation."""
    return _unstack_pytree(self._stacked_post_processed_outputs)

  @functools.cached_property
  def _edge_outputs(self) -> Sequence[edge_base.EdgeModelOutputs | None]:
    """Returns the edge outputs for the simulation."""
    if self._stacked_edge_outputs is None:
      return (None,) * len(self.times)
    return _unstack_pytree(self._stacked_edge_outputs)

  def _build_top_level_dataset(self) -> xr.Dataset:
    """Builds the top-level xr.Dataset with coordinates and config metadata."""
    time = xr.DataArray(
        np.asarray(self.times),
        dims=[output_keys.TIME],
        name=output_keys.TIME,
        attrs=output_keys.get_units(output_keys.TIME),
    )
    rho_face_norm = xr.DataArray(
        np.asarray(self.rho_face_norm),
        dims=[output_keys.RHO_FACE_NORM],
        name=output_keys.RHO_FACE_NORM,
        attrs=output_keys.get_units(output_keys.RHO_FACE_NORM),
    )
    rho_cell_norm = xr.DataArray(
        np.asarray(self.rho_cell_norm),
        dims=[output_keys.RHO_CELL_NORM],
        name=output_keys.RHO_CELL_NORM,
        attrs=output_keys.get_units(output_keys.RHO_CELL_NORM),
    )
    rho_norm = xr.DataArray(
        np.asarray(self.rho_norm),
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
    return xr.Dataset(
        data_vars=None,
        coords=coords,
        attrs={output_keys.CONFIG: self.torax_config.model_dump_json()},
    )

  def _build_profiles_and_scalars_datasets(
      self,
  ) -> tuple[xr.Dataset, xr.Dataset]:
    """Builds the profiles and scalars xr.Datasets from stacked state dicts."""
    grid = self._output_grid_context
    output_dicts = [
        self._core_profiles.to_output_dict(grid),
        self._stacked_core_transport.to_output_dict(grid),
        self._core_sources.to_output_dict(grid),
        self._stacked_post_processed_outputs.to_output_dict(grid),
        self._geometry.to_output_dict(grid),
    ]
    all_vars: dict[str, output_grid_context.OutputVar] = {}
    for d in output_dicts:
      duplicate_keys = set(all_vars).intersection(d)
      if duplicate_keys:
        raise ValueError(
            f"Duplicate keys found in output dicts: {duplicate_keys}"
        )
      all_vars.update(d)

    spatial_coords = {
        output_keys.RHO_FACE_NORM,
        output_keys.RHO_CELL_NORM,
        output_keys.RHO_NORM,
    }
    profiles_vars = {
        k: v
        for k, v in all_vars.items()
        if any(d in spatial_coords for d in v[0])
    }
    scalars_vars = {
        k: v
        for k, v in all_vars.items()
        if not any(d in spatial_coords for d in v[0])
    }
    return (
        grid.build_dataset(profiles_vars),
        grid.build_dataset(scalars_vars),
    )

  def _build_numerics_dataset(self) -> xr.Dataset:
    """Builds the numerics xr.Dataset."""
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
    numerics_vars.update(
        self._stacked_solver_numeric_outputs.to_output_dict(
            self._output_grid_context
        )
    )
    return self._output_grid_context.build_dataset(numerics_vars)

  def _build_auxiliary_datatree(self) -> xr.DataTree | None:
    """Builds the auxiliary xr.DataTree if turbulent transport has children."""
    turbulent_transport_tree = (
        self._stacked_core_transport.turbulent.to_xr_datatree(
            self._output_grid_context
        )
    )
    if not turbulent_transport_tree.children:
      return None
    return xr.DataTree(
        children={
            output_keys.TURBULENT_TRANSPORT: turbulent_transport_tree,
        },
    )

  def simulation_output_to_xr(self) -> xr.DataTree:
    """Build an xr.DataTree of the simulation output.

    This method converts JAX arrays to NumPy arrays and cannot be called under
    `jax.jit`.

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
        - auxiliary: Contains auxiliary model outputs, such as turbulent
            transport component outputs, if any are present.
    """
    profiles_dataset, scalars_dataset = (
        self._build_profiles_and_scalars_datasets()
    )
    children: dict[str, xr.DataTree] = {
        output_keys.PROFILES: xr.DataTree(dataset=profiles_dataset),
        output_keys.SCALARS: xr.DataTree(dataset=scalars_dataset),
        output_keys.NUMERICS: xr.DataTree(
            dataset=self._build_numerics_dataset()
        ),
    }
    if self._stacked_edge_outputs is not None:
      children[output_keys.EDGE] = self._stacked_edge_outputs.to_xr_datatree(
          self._output_grid_context
      )

    auxiliary_tree = self._build_auxiliary_datatree()
    if auxiliary_tree is not None:
      children[output_keys.AUXILIARY] = auxiliary_tree

    data_tree = xr.DataTree(
        children=children,
        dataset=self._build_top_level_dataset(),
    )
    if (
        self.torax_config.restart is not None
        and self.torax_config.restart.stitch
    ):
      data_tree = stitch_state_files(self.torax_config.restart, data_tree)

    return data_tree
