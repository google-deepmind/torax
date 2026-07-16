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

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import inspect
import itertools
from typing import cast

import os

from absl import logging
import chex
import jax
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.edge import base as edge_base
from torax._src.edge import extended_lengyel_standalone
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import output_keys
from torax._src.output_tools import post_processing
from torax._src.solver import jax_root_finding
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import model_config
import xarray as xr

# Internal import.

# pylint: disable=invalid-name

# CoreProfiles field names excluded from direct output serialization.
# These are either redundant, require special handling, or belong elsewhere.
_EXCLUDED_CORE_PROFILE_FIELDS = frozenset({
    "impurity_fractions",  # Redundant with n_impurity_species.
    "charge_state_info",
    "charge_state_info_face",
    "impurity_density_scaling",
    "fast_ions",  # Handled separately via fast ion loop.
    "n_impurity_thermal",
    "main_ion_fractions",  # Requires special handling (dict → DataArray).
    # TODO(b/434175938): Remove once we move to V2.
    "internal_plasma_energy",  # In post_processed_outputs.
})

# Geometry field names excluded from direct output serialization.
_EXCLUDED_GEOMETRY_FIELDS = frozenset({
    "geometry_type",
    "Ip_from_parameters",
    "j_total",  # Already in core_profiles output.
})

# Geometry property names excluded because they are coordinates at the top
# DataTree level, or are not interesting to save.
# TODO(b/338033916): consolidate on either rho or rho_cell naming for cell grid
_EXCLUDED_GEOMETRY_PROPERTIES = frozenset({
    output_keys.RHO_FACE,
    output_keys.RHO_CELL,
    output_keys.RHO_CELL_NORM,
    output_keys.RHO_FACE_NORM,
    "rho",
    "rho_norm",
    "q_correction_factor",
})


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


def extend_cell_grid_to_boundaries(
    cell_var: array_typing.FloatVectorCell,
    face_var: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorCellPlusBoundaries:
  """Merge face+cell grids into single [left_face, cells, right_face] grid."""

  left_value = np.expand_dims(face_var[:, 0], axis=-1)
  right_value = np.expand_dims(face_var[:, -1], axis=-1)

  return np.concatenate([left_value, cell_var, right_value], axis=-1)


def stitch_state_files(
    file_restart: file_restart_pydantic_model.FileRestart, datatree: xr.DataTree
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
    self._stacked_geometry = geometry_lib.stack_geometries(self.geometries)
    stack = lambda *ys: np.stack(ys)
    self._stacked_core_profiles: state.CoreProfiles = jax.tree_util.tree_map(
        stack, *self._core_profiles
    )
    self._stacked_core_sources: source_profiles_lib.SourceProfiles = (
        jax.tree_util.tree_map(stack, *self._core_sources)
    )
    self._stacked_core_transport: state.CoreTransport = jax.tree_util.tree_map(
        stack, *self._transport
    )
    self._stacked_post_processed_outputs: (
        post_processing.PostProcessedOutputs
    ) = jax.tree_util.tree_map(stack, *post_processed_outputs_history)
    self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
        jax.tree_util.tree_map(stack, *self._solver_numeric_outputs)
    )
    # If self._edge_outputs is a list of Nones, stacked_edge_outputs will just
    # be None, since jax.tree_util.tree_map treats None as a node with no leaves
    self._stacked_edge_outputs: edge_base.EdgeModelOutputs = (
        jax.tree_util.tree_map(stack, *self._edge_outputs)
    )

    self._times = np.array([state.t for state in state_history])
    chex.assert_rank(self.times, 1)
    # The rho grid does not change in time so we can just take the first one.
    self._rho_cell_norm = state_history[0].geometry.rho_norm
    self._rho_face_norm = state_history[0].geometry.rho_face_norm
    self._rho_norm = np.concatenate([[0.0], self.rho_cell_norm, [1.0]])

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
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

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

    # Update dict with flattened StateHistory dataclass containers
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

    # Determine simulation status based on error state
    sim_status = (
        state.SimStatus.COMPLETED
        if self.sim_error is state.SimError.NO_ERROR
        else state.SimStatus.ERROR
    )

    numerics_dict = {
        output_keys.SIM_STATUS: sim_status.value,
        output_keys.SIM_ERROR: self.sim_error.value,
        output_keys.SAWTOOTH_CRASH: xr.DataArray(
            self._stacked_solver_numeric_outputs.sawtooth_crash,
            dims=[output_keys.TIME],
            name=output_keys.SAWTOOTH_CRASH,
        ),
        output_keys.OUTER_SOLVER_ITERATIONS: xr.DataArray(
            self._stacked_solver_numeric_outputs.outer_solver_iterations,
            dims=[output_keys.TIME],
            name=output_keys.OUTER_SOLVER_ITERATIONS,
        ),
        output_keys.INNER_SOLVER_ITERATIONS: xr.DataArray(
            self._stacked_solver_numeric_outputs.inner_solver_iterations,
            dims=[output_keys.TIME],
            name=output_keys.INNER_SOLVER_ITERATIONS,
        ),
    }
    numerics = xr.Dataset(numerics_dict)

    spatial_coords = {
        output_keys.RHO_FACE_NORM,
        output_keys.RHO_CELL_NORM,
        output_keys.RHO_NORM,
    }

    profiles_dict = {
        k: v
        for k, v in flattened_all_core_data.items()
        if v is not None and any(d in spatial_coords for d in v.dims)  # pytype: disable=attribute-error
    }
    profiles = xr.Dataset(profiles_dict)
    scalars_dict = {
        k: v
        for k, v in flattened_all_core_data.items()
        if v is not None and not any(d in spatial_coords for d in v.dims)  # pytype: disable=attribute-error
    }
    scalars = xr.Dataset(scalars_dict)
    children = {
        output_keys.NUMERICS: xr.DataTree(dataset=numerics),
        output_keys.PROFILES: xr.DataTree(dataset=profiles),
        output_keys.SCALARS: xr.DataTree(dataset=scalars),
    }
    if self._stacked_edge_outputs is not None:
      children[output_keys.EDGE] = self._save_edge_outputs()
    data_tree = xr.DataTree(
        children=children,
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
        len(self.rho_cell_norm),
    )
    is_cell_plus_boundaries_var = lambda x: x.ndim == 2 and x.shape == (
        len(self.times),
        len(self.rho_norm),
    )
    is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times),)
    is_constant = lambda x: x.ndim == 0

    match data:
      case data if is_face_var(data):
        dims = [output_keys.TIME, output_keys.RHO_FACE_NORM]
      case data if is_cell_var(data):
        dims = [output_keys.TIME, output_keys.RHO_CELL_NORM]
      case data if is_scalar(data):
        dims = [output_keys.TIME]
      case data if is_constant(data):
        dims = []
      case data if is_cell_plus_boundaries_var(data):
        dims = [output_keys.TIME, output_keys.RHO_NORM]
      case _:
        logging.warning(
            "Unsupported data shape for %s: %s. Skipping persisting.",
            name,
            data.shape,  # pytype: disable=attribute-error
        )
        return None

    return xr.DataArray(
        data, dims=dims, name=name, attrs=output_keys.get_units(name)
    )

  def _save_core_profiles(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the stacked core profiles to a dictionary of xr.DataArrays."""
    xr_dict = {}
    stacked_core_profiles = self._stacked_core_profiles

    # Map from CoreProfiles attribute name to the desired output name.
    # Needed for attributes that are not 1:1 with the output name.
    # Other attributes will use the same name as in CoreProfiles
    output_name_map = {
        "psidot": output_keys.V_LOOP,
        "sigma": output_keys.SIGMA_PARALLEL,
        "Ip_profile_face": output_keys.IP_PROFILE,
        "q_face": output_keys.Q,
        "s_face": output_keys.MAGNETIC_SHEAR,
    }

    core_profile_field_names = {
        f.name for f in dataclasses.fields(stacked_core_profiles)
    }

    # Add cached_properties to the list of fields to save.
    core_profiles_cached_properties = inspect.getmembers(
        type(stacked_core_profiles),
        lambda member: isinstance(member, functools.cached_property),
    )
    core_profiles_cached_properties_names = set(
        [name for name, _ in core_profiles_cached_properties]
    )

    core_profiles_names = (
        core_profile_field_names | core_profiles_cached_properties_names
    )

    for attr_name in core_profiles_names:
      if attr_name in _EXCLUDED_CORE_PROFILE_FIELDS:
        continue

      attr_value = getattr(stacked_core_profiles, attr_name)

      output_key = output_name_map.get(attr_name, attr_name)

      # Skip _face attributes if their cell counterpart exists;
      # they are handled when the cell attribute is processed.
      if attr_name.endswith("_face") and (
          attr_name.removesuffix("_face") in core_profiles_names
      ):
        continue

      # Special handling for A_impurity for backward compatibility with V1
      # API for default 'fractions' impurity mode where A_impurity was a scalar.
      # TODO(b/434175938): Remove this once we move to V2
      if attr_name == "A_impurity":
        # Check if A_impurity is constant across the radial dimension for all
        # time steps. Need slicing (not indexing) to avoid a broadcasting error.
        is_constant = np.all(attr_value == attr_value[..., 0:1], axis=-1)
        if np.all(is_constant):
          # Save as a scalar time-series. Take the value at the first point.
          data_to_save = attr_value[..., 0]
        else:
          # Save as a profile.
          face_value = getattr(stacked_core_profiles, "A_impurity_face")
          data_to_save = extend_cell_grid_to_boundaries(attr_value, face_value)
        xr_dict[output_key] = self._pack_into_data_array(
            output_key, data_to_save  # pyrefly: ignore[bad-argument-type]
        )
        continue

      if isinstance(attr_value, cell_variable.CellVariable):
        # Handles stacked CellVariable-like objects.
        data_to_save = []
        for core_profile in self.core_profiles:
          cell_var: cell_variable.CellVariable = getattr(
              core_profile, attr_name
          )
          data_to_save.append(cell_var.cell_plus_boundaries())
        data_to_save = np.stack(data_to_save)
      else:
        face_attr_name = f"{attr_name}_face"
        if face_attr_name in core_profile_field_names:
          # Combine cell and edge face values.
          face_value = getattr(stacked_core_profiles, face_attr_name)
          data_to_save = extend_cell_grid_to_boundaries(attr_value, face_value)
        else:  # cell array with no face counterpart, or a scalar value
          data_to_save = attr_value

      xr_dict[output_key] = self._pack_into_data_array(output_key, data_to_save)  # pyrefly: ignore[bad-argument-type]

    # Handle derived quantities
    Ip_data = stacked_core_profiles.Ip_profile_face[..., -1]
    xr_dict[output_keys.IP] = self._pack_into_data_array(output_keys.IP, Ip_data)  # pyrefly: ignore[bad-argument-type]

    # Handle main_ion_fractions
    main_ions = sorted(list(stacked_core_profiles.main_ion_fractions.keys()))
    data = np.stack(
        [stacked_core_profiles.main_ion_fractions[ion] for ion in main_ions],
        axis=0,
    )
    xr_dict[output_keys.MAIN_ION_FRACTIONS] = xr.DataArray(
        data,
        dims=[output_keys.MAIN_ION, output_keys.TIME],
        coords={
            output_keys.MAIN_ION: main_ions,
            output_keys.TIME: self.times,
        },
        name=output_keys.MAIN_ION_FRACTIONS,
        attrs=output_keys.get_units(output_keys.MAIN_ION_FRACTIONS),
    )
    # Handle fast ions
    first_fast_ions = self.core_profiles[0].fast_ions
    for i, first_fi in enumerate(first_fast_ions):
      source_key = f"{first_fi.source}_{first_fi.species}"
      n_data = np.stack([
          cp.fast_ions[i].n.cell_plus_boundaries() for cp in self.core_profiles
      ])
      T_data = np.stack([
          cp.fast_ions[i].T.cell_plus_boundaries() for cp in self.core_profiles
      ])
      n_key = output_keys.n_fast_ion_key(source_key)
      T_key = output_keys.T_fast_ion_key(source_key)
      xr_dict[n_key] = self._pack_into_data_array(
          n_key, n_data  # pyrefly: ignore[bad-argument-type]
      )
      xr_dict[T_key] = self._pack_into_data_array(
          T_key, T_data  # pyrefly: ignore[bad-argument-type]
      )

    return xr_dict

  def _save_core_transport(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core transport to a dict."""
    xr_dict = {}
    core_transport = self._stacked_core_transport

    xr_dict[output_keys.CHI_TURB_I] = core_transport.chi_face_ion
    xr_dict[output_keys.CHI_TURB_E] = core_transport.chi_face_el
    xr_dict[output_keys.D_TURB_E] = core_transport.d_face_el
    xr_dict[output_keys.V_TURB_E] = core_transport.v_face_el

    xr_dict[output_keys.CHI_NEO_I] = core_transport.chi_neo_i
    xr_dict[output_keys.CHI_NEO_E] = core_transport.chi_neo_e
    xr_dict[output_keys.D_NEO_E] = core_transport.D_neo_e
    xr_dict[output_keys.V_NEO_E] = core_transport.V_neo_e
    xr_dict[output_keys.V_NEO_WARE_E] = core_transport.V_neo_ware_e

    # Save optional BohmGyroBohm attributes if present.
    core_transport = self._stacked_core_transport
    optional_transport_map = {
        output_keys.CHI_BOHM_E: core_transport.chi_face_el_bohm,
        output_keys.CHI_GYROBOHM_E: core_transport.chi_face_el_gyrobohm,
        output_keys.CHI_BOHM_I: core_transport.chi_face_ion_bohm,
        output_keys.CHI_GYROBOHM_I: core_transport.chi_face_ion_gyrobohm,
        output_keys.CHI_ITG_E: core_transport.chi_face_el_itg,
        output_keys.CHI_TEM_E: core_transport.chi_face_el_tem,
        output_keys.CHI_ETG_E: core_transport.chi_face_el_etg,
        output_keys.CHI_ITG_I: core_transport.chi_face_ion_itg,
        output_keys.CHI_TEM_I: core_transport.chi_face_ion_tem,
        output_keys.D_ITG_E: core_transport.d_face_el_itg,
        output_keys.D_TEM_E: core_transport.d_face_el_tem,
        output_keys.V_ITG_E: core_transport.v_face_el_itg,
        output_keys.V_TEM_E: core_transport.v_face_el_tem,
    }

    for name, data in optional_transport_map.items():
      # Skip if None or an array of Nones from stack
      if data is not None and data.dtype != object:
        xr_dict[name] = data

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

    xr_dict[qei_source_lib.QeiSource.SOURCE_NAME] = (
        self._stacked_core_sources.qei.qei_coef
        * (
            self._stacked_core_profiles.T_e.value  # pyrefly: ignore[unsupported-operation]
            - self._stacked_core_profiles.T_i.value
        )
    )

    xr_dict[output_keys.J_PARALLEL_BOOTSTRAP] = extend_cell_grid_to_boundaries(
        self._stacked_core_sources.bootstrap_current.j_parallel_bootstrap,
        self._stacked_core_sources.bootstrap_current.j_parallel_bootstrap_face,
    )

    # Add source profiles with suffixes indicating which profile they affect.
    for profile in self._stacked_core_sources.T_i:
      name = output_keys.SOURCE_NAME_RENAMES.get(profile, profile)
      key = output_keys.p_source_i_key(name)
      xr_dict[key] = self._stacked_core_sources.T_i[profile]
    for profile in self._stacked_core_sources.T_e:
      name = output_keys.SOURCE_NAME_RENAMES.get(profile, profile)
      key = output_keys.p_source_e_key(name)
      xr_dict[key] = self._stacked_core_sources.T_e[profile]
    for profile in self._stacked_core_sources.psi:
      key = output_keys.j_parallel_source_key(profile)
      xr_dict[key] = self._stacked_core_sources.psi[profile]
    for profile in self._stacked_core_sources.n_e:
      key = output_keys.s_source_key(profile)
      xr_dict[key] = self._stacked_core_sources.n_e[profile]

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

    pp_field_names = {
        f.name for f in dataclasses.fields(self._stacked_post_processed_outputs)
    }

    for field in dataclasses.fields(self._stacked_post_processed_outputs):
      attr_name = field.name
      if attr_name == "first_step":
        continue

      # The impurity_radiation is structured differently and handled separately.
      if attr_name == "impurity_species":
        continue

      # Skip _face attributes if cell counterpart exists
      if attr_name.endswith("_face") and (
          attr_name.removesuffix("_face") in pp_field_names
      ):
        continue

      attr_value = getattr(self._stacked_post_processed_outputs, attr_name)

      # Check if a corresponding face variable exists to extend the grid.
      face_attr_name = f"{attr_name}_face"

      if hasattr(attr_value, "cell_plus_boundaries"):
        # Handles stacked CellVariable-like objects.
        data_to_save = attr_value.cell_plus_boundaries()
      elif face_attr_name in pp_field_names:
        face_value = getattr(
            self._stacked_post_processed_outputs, face_attr_name
        )
        data_to_save = extend_cell_grid_to_boundaries(attr_value, face_value)
      else:
        data_to_save = attr_value
      xr_dict[attr_name] = self._pack_into_data_array(attr_name, data_to_save)  # pyrefly: ignore[bad-argument-type]

    if self._stacked_post_processed_outputs.impurity_species:
      radiation_outputs = (
          impurity_radiation.construct_xarray_for_radiation_output(
              self._stacked_post_processed_outputs.impurity_species,
              self.times,  # pyrefly: ignore[bad-argument-type]
              self.rho_cell_norm,  # pyrefly: ignore[bad-argument-type]
              output_keys.TIME,
              output_keys.RHO_CELL_NORM,
          )
      )
      for key, value in radiation_outputs.items():
        xr_dict[key] = value

    return xr_dict

  def _save_geometry(
      self,
  ) -> dict[str, xr.DataArray]:
    """Save geometry to a dict. We skip over hires and non-array quantities."""
    xr_dict = {}
    geometry_attributes = dataclasses.asdict(self._stacked_geometry)

    # Get the variables from dataclass fields.
    for field_name, data in geometry_attributes.items():
      if (
          "hires" in field_name
          or (
              field_name.endswith("_face")
              and field_name.removesuffix("_face") in geometry_attributes
          )
          or field_name in _EXCLUDED_GEOMETRY_FIELDS
          or not isinstance(data, array_typing.Array)
      ):
        continue
      if f"{field_name}_face" in geometry_attributes:
        data = extend_cell_grid_to_boundaries(
            data, geometry_attributes[f"{field_name}_face"]
        )
      # Remap to avoid outputting _face suffix in output.
      if field_name.endswith("_face"):
        field_name = field_name.removesuffix("_face")
      if field_name == "Ip_profile":
        # Ip_profile exists in core profiles so rename to avoid duplicate.
        field_name = output_keys.IP_PROFILE_FROM_GEO
      if field_name == "psi":
        # Psi also exists in core profiles so rename to avoid duplicate.
        field_name = output_keys.PSI_FROM_GEO
      if field_name == "_z_magnetic_axis":
        # This logic only reached if not None. Avoid leading underscore.
        field_name = output_keys.Z_MAGNETIC_AXIS
      data_array = self._pack_into_data_array(
          field_name,
          data,  # pyrefly: ignore[bad-argument-type]
      )
      if data_array is not None:
        xr_dict[field_name] = data_array

    # Get variables from property methods
    geometry_properties = inspect.getmembers(type(self._stacked_geometry))
    property_names = set([name for name, _ in geometry_properties])

    for name, value in geometry_properties:
      # Skip over saving any variables that are named *_face.
      if (
          name.endswith("_face")
          and name.removesuffix("_face") in property_names
      ):
        continue
      if name in _EXCLUDED_GEOMETRY_PROPERTIES:
        continue
      if isinstance(value, property):
        property_data = value.fget(self._stacked_geometry)  # pyrefly: ignore[not-callable]
        # TODO(b/434175938): Remove this once we move to V2.
        if name == "drho":
          is_uniform = np.all(
              np.isclose(property_data[:, 0][..., None], property_data)
          )
          if is_uniform:
            property_data = property_data[:, 0]
        elif name == "drho_norm":
          is_uniform = all(np.isclose(property_data, property_data[0]))
          if is_uniform:
            property_data = property_data[0]
        # Check if there is a corresponding face variable for this property.
        # If so, extend the data to the cell+boundaries grid.
        if f"{name}_face" in property_names:
          face_data = getattr(self._stacked_geometry, f"{name}_face")
          property_data = extend_cell_grid_to_boundaries(
              property_data, face_data
          )
        # Remap to avoid outputting _face suffix in output. Done only for
        # _face variables with no corresponding non-face variable.
        if name.endswith("_face"):
          name = name.removesuffix("_face")
        data_array = self._pack_into_data_array(name, property_data)  # pyrefly: ignore[bad-argument-type]
        if data_array is not None:
          xr_dict[name] = data_array

    return xr_dict

  def _save_edge_outputs(self) -> xr.DataTree:
    """Saves the edge outputs to a DataTree."""
    xr_dict = {}
    children = {}
    outputs = self._stacked_edge_outputs

    # Fields from ExtendedLengyelOutputs
    # TODO(b/446608829): generalize when additional edge models are added
    if not isinstance(
        outputs, extended_lengyel_standalone.ExtendedLengyelOutputs
    ):
      # Return empty DataTree for non-extended-lengyel edge outputs.
      return xr.DataTree(dataset=xr.Dataset(xr_dict))

    standard_output_fields = [
        output_keys.Q_PARALLEL,
        output_keys.Q_PERPENDICULAR_TARGET,
        output_keys.T_E_SEPARATRIX,
        output_keys.T_E_TARGET,
        output_keys.PRESSURE_NEUTRAL_DIVERTOR,
        output_keys.ALPHA_T,
        output_keys.Z_EFF_SEPARATRIX,
        output_keys.MULTIPLE_ROOTS_FOUND,
    ]

    edge_output_fields = dataclasses.fields(outputs)
    for field in edge_output_fields:
      name = field.name
      value = getattr(outputs, name)
      if field.name in standard_output_fields:
        packed = self._pack_into_data_array(name, value)
        if packed is not None:
          xr_dict[name] = packed
        continue
      # Special handling for seed_impurity_concentrations
      if name == output_keys.SEED_IMPURITY_CONCENTRATIONS and value:
        # This is a dict of {impurity: array(time,)}, where (time,) is the shape
        # We want to convert it to an array of shape (n_impurities, time) with
        # impurity coord.
        impurities = sorted(list(value.keys()))
        data_array = np.stack([value[i] for i in impurities], axis=0)
        xr_dict[name] = xr.DataArray(
            data_array,
            dims=[output_keys.SEED_IMPURITY, output_keys.TIME],
            coords={
                output_keys.SEED_IMPURITY: impurities,
                output_keys.TIME: self.times,
            },
            name=name,
            attrs=output_keys.get_units(name),
        )
        continue

      if name == output_keys.CALCULATED_ENRICHMENT and value:
        # This is a dict of {impurity: array(time,)}, where (time,) is the shape
        # We want to convert it to an array of shape (n_impurities, time) with
        # impurity coord.
        impurities = sorted(list(value.keys()))
        data_array = np.stack([value[i] for i in impurities], axis=0)
        xr_dict[name] = xr.DataArray(
            data_array,
            dims=[output_keys.IMPURITY, output_keys.TIME],
            coords={
                output_keys.IMPURITY: impurities,
                output_keys.TIME: self.times,
            },
            name=name,
        )
        continue

      if name == output_keys.ROOTS and value is not None:
        roots_dict = self._process_roots_output(outputs)
        if roots_dict:
          children[output_keys.ROOTS] = xr.DataTree(
              dataset=xr.Dataset(roots_dict)
          )

    # Fields from SolverStatus which depend on the solver type
    xr_dict[output_keys.SOLVER_PHYSICS_OUTCOME] = self._pack_into_data_array(
        output_keys.SOLVER_PHYSICS_OUTCOME, outputs.solver_status.physics_outcome  # pyrefly: ignore[bad-argument-type]
    )
    numerics = outputs.solver_status.numerics_outcome
    # Check for RootMetadata structure (newton solver)
    # TODO(b/446608829): make numerics itself parse its contents for outputs.
    if isinstance(numerics, jax_root_finding.RootMetadata):
      xr_dict[output_keys.SOLVER_ITERATIONS] = self._pack_into_data_array(
          output_keys.SOLVER_ITERATIONS, numerics.iterations
      )
      # Handle solver_residual explicitly because it is a vector
      # (time, n_unknowns). We want to output the scalar metric used for
      # convergence checking (mean absolute error).
      residual_scalar = np.mean(np.abs(numerics.residual), axis=-1)
      xr_dict[output_keys.SOLVER_RESIDUAL] = self._pack_into_data_array(
          output_keys.SOLVER_RESIDUAL, residual_scalar
      )

      xr_dict[output_keys.SOLVER_ERROR] = self._pack_into_data_array(
          output_keys.SOLVER_ERROR, numerics.error
      )
    else:
      # FixedPointOutcome (fixed point solver)
      xr_dict[output_keys.FIXED_POINT_OUTCOME] = self._pack_into_data_array(
          output_keys.FIXED_POINT_OUTCOME, numerics  # pyrefly: ignore[bad-argument-type]
      )

    return xr.DataTree(dataset=xr.Dataset(xr_dict), children=children)

  def _process_roots_output(
      self, roots: extended_lengyel_standalone.ExtendedLengyelOutputs
  ) -> Mapping[str, xr.DataArray]:
    """Processes roots output to identify unique valid roots."""
    unique_roots_obj = roots.get_unique_roots()
    if unique_roots_obj is None:
      return {}

    xr_dict = {}
    default_dims = [output_keys.TIME, output_keys.N_ROOTS]

    # Helper to add data to xr_dict with correct dims
    def _add_to_xr(name: str, data: jax.Array):
      if data.ndim == len(default_dims):
        dims = default_dims
      elif data.ndim > len(default_dims):
        # Handle extra dimensions (e.g. vector residual)
        extra_dims = [
            f"{name}_dim_{i}" for i in range(data.ndim - len(default_dims))
        ]
        dims = default_dims + extra_dims
      else:
        dims = default_dims[: data.ndim]

      xr_dict[name] = xr.DataArray(data, dims=dims, name=name)

    # Iterate over fields of the returned object
    # Cast to concrete type for Pytype
    unique_roots_obj = cast(
        extended_lengyel_standalone.ExtendedLengyelOutputs, unique_roots_obj
    )
    for field in dataclasses.fields(unique_roots_obj):
      name = field.name
      # Skip internal fields or recursion or solver_status (handled explicitly)
      if name in (
          output_keys.ROOTS,
          output_keys.MULTIPLE_ROOTS_FOUND,
          "solver_status",
      ):
        continue

      value = getattr(unique_roots_obj, name)

      if isinstance(value, (jax.Array, np.ndarray)):
        _add_to_xr(name, value)  # pyrefly: ignore[bad-argument-type]
      elif isinstance(value, Mapping):
        for k, v in value.items():
          # Flatten dict fields -> name_key
          _add_to_xr(f"{name}_{k}", v)

    # Handle solver_status explicitly to flatten metrics
    status = unique_roots_obj.solver_status
    if status.physics_outcome is not None:
      _add_to_xr(
          output_keys.SOLVER_PHYSICS_OUTCOME,
          jax.numpy.asarray(status.physics_outcome),
      )

    numerics = status.numerics_outcome
    if isinstance(numerics, jax_root_finding.RootMetadata):
      _add_to_xr(output_keys.SOLVER_ITERATIONS, numerics.iterations)
      _add_to_xr(output_keys.SOLVER_RESIDUAL, numerics.residual)
      _add_to_xr(output_keys.SOLVER_ERROR, numerics.error)

    return xr_dict
