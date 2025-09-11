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
import inspect
import itertools

import os

from absl import logging
import chex
import jax
import numpy as np
import os
from torax._src import array_typing
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import post_processing
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import model_config
import xarray as xr

# pylint: disable=invalid-name

# Dataset names.
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"

# Core profiles.
T_E = "T_e"
T_I = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
MAGNETIC_SHEAR = "magnetic_shear"
N_IMPURITY = "n_impurity"
Z_IMPURITY = "Z_impurity"
Z_EFF = "Z_eff"
SIGMA_PARALLEL = "sigma_parallel"
V_LOOP_LCFS = "v_loop_lcfs"
J_TOTAL = "j_total"
IP_PROFILE = "Ip_profile"
IP = "Ip"

# Calculated or derived currents.
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"

# Core transport.
CHI_TURB_I = "chi_turb_i"
CHI_TURB_E = "chi_turb_e"
D_TURB_E = "D_turb_e"
V_TURB_E = "V_turb_e"
CHI_NEO_I = "chi_neo_i"
CHI_NEO_E = "chi_neo_e"
D_NEO_E = "D_neo_e"
V_NEO_E = "V_neo_e"
V_NEO_WARE_E = "V_neo_ware_e"
CHI_BOHM_E = "chi_bohm_e"
CHI_GYROBOHM_E = "chi_gyrobohm_e"
CHI_BOHM_I = "chi_bohm_i"
CHI_GYROBOHM_I = "chi_gyrobohm_i"

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_NORM = "rho_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Post processed outputs
Q_FUSION = "Q_fusion"

# Numerics.
# Simulation error state.
SIM_ERROR = "sim_error"
OUTER_SOLVER_ITERATIONS = "outer_solver_iterations"
INNER_SOLVER_ITERATIONS = "inner_solver_iterations"
# Boolean array indicating whether the state corresponds to a
# post-sawtooth-crash state.
SAWTOOTH_CRASH = "sawtooth_crash"

# ToraxConfig.
CONFIG = "config"

# Excluded coordinates from geometry since they are at the top DataTree level.
# Exclude q_correction_factor as it is not an interesting quantity to save.
# TODO(b/338033916): consolidate on either rho or rho_cell naming for cell grid
EXCLUDED_GEOMETRY_NAMES = frozenset({
    RHO_FACE,
    RHO_CELL,
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


def _extend_cell_grid_to_boundaries(
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
      state_history: tuple[sim_state.ToraxSimState, ...],
      post_processed_outputs_history: tuple[
          post_processing.PostProcessedOutputs, ...
      ],
      sim_error: state.SimError,
      torax_config: model_config.ToraxConfig,
  ):
    if (
        not torax_config.restart
        and not torax_config.profile_conditions.use_v_loop_lcfs_boundary_condition
        and len(state_history) >= 2
    ):
      # For the Ip BC case, set v_loop_lcfs[0] to the same value as
      # v_loop_lcfs[1] due the v_loop_lcfs timeseries being
      # underconstrained
      state_history[0].core_profiles = dataclasses.replace(
          state_history[0].core_profiles,
          v_loop_lcfs=state_history[1].core_profiles.v_loop_lcfs,
      )
    self._sim_error = sim_error
    self._torax_config = torax_config
    self._post_processed_outputs = post_processed_outputs_history
    self._solver_numeric_outputs = [
        state.solver_numeric_outputs for state in state_history
    ]
    self._core_profiles = [state.core_profiles for state in state_history]
    self._core_sources = [state.core_sources for state in state_history]
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
    """
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    # Get coordinate variables for dimensions ("time", "rho_face", "rho_cell")
    time = xr.DataArray(self.times, dims=[TIME], name=TIME)
    rho_face_norm = xr.DataArray(
        self.rho_face_norm, dims=[RHO_FACE_NORM], name=RHO_FACE_NORM
    )
    rho_cell_norm = xr.DataArray(
        self.rho_cell_norm, dims=[RHO_CELL_NORM], name=RHO_CELL_NORM
    )
    rho_norm = xr.DataArray(
        self.rho_norm,
        dims=[RHO_NORM],
        name=RHO_NORM,
    )

    coords = {
        TIME: time,
        RHO_FACE_NORM: rho_face_norm,
        RHO_CELL_NORM: rho_cell_norm,
        RHO_NORM: rho_norm,
    }

    # Update dict with flattened StateHistory dataclass containers
    all_dicts = [
        self._save_core_profiles(),
        self._save_core_transport(),
        self._save_core_sources(),
        self._save_post_processed_outputs(),
        self._save_geometry(),
    ]
    flat_dict = {}
    for key, value in itertools.chain(*(d.items() for d in all_dicts)):
      if key not in flat_dict:
        flat_dict[key] = value
      else:
        raise ValueError(f"Duplicate key: {key}")
    numerics_dict = {
        SIM_ERROR: self.sim_error.value,
        SAWTOOTH_CRASH: xr.DataArray(
            self._stacked_solver_numeric_outputs.sawtooth_crash,
            dims=[TIME],
            name=SAWTOOTH_CRASH,
        ),
        OUTER_SOLVER_ITERATIONS: xr.DataArray(
            self._stacked_solver_numeric_outputs.outer_solver_iterations,
            dims=[TIME],
            name=OUTER_SOLVER_ITERATIONS,
        ),
        INNER_SOLVER_ITERATIONS: xr.DataArray(
            self._stacked_solver_numeric_outputs.inner_solver_iterations,
            dims=[TIME],
            name=INNER_SOLVER_ITERATIONS,
        ),
    }
    numerics = xr.Dataset(numerics_dict)
    profiles_dict = {
        k: v
        for k, v in flat_dict.items()
        if v is not None and v.values.ndim > 1  # pytype: disable=attribute-error
    }
    profiles = xr.Dataset(profiles_dict)
    scalars_dict = {
        k: v
        for k, v in flat_dict.items()
        if v is not None and v.values.ndim in [0, 1]  # pytype: disable=attribute-error
    }
    scalars = xr.Dataset(scalars_dict)
    data_tree = xr.DataTree(
        children={
            NUMERICS: xr.DataTree(dataset=numerics),
            PROFILES: xr.DataTree(dataset=profiles),
            SCALARS: xr.DataTree(dataset=scalars),
        },
        dataset=xr.Dataset(
            data_vars=None,
            coords=coords,
            attrs={CONFIG: self.torax_config.model_dump_json()},
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
        dims = [TIME, RHO_FACE_NORM]
      case data if is_cell_var(data):
        dims = [TIME, RHO_CELL_NORM]
      case data if is_scalar(data):
        dims = [TIME]
      case data if is_constant(data):
        dims = []
      case data if is_cell_plus_boundaries_var(data):
        dims = [TIME, RHO_NORM]
      case _:
        logging.warning(
            "Unsupported data shape for %s: %s. Skipping persisting.",
            name,
            data.shape,  # pytype: disable=attribute-error
        )
        return None

    return xr.DataArray(data, dims=dims, name=name)

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
        "psidot": V_LOOP,
        "sigma": SIGMA_PARALLEL,
        "Ip_profile_face": IP_PROFILE,
        "q_face": Q,
        "s_face": MAGNETIC_SHEAR,
    }

    core_profile_field_names = {
        f.name for f in dataclasses.fields(stacked_core_profiles)
    }

    for field in dataclasses.fields(stacked_core_profiles):
      attr_name = field.name

      # Skip impurity_fractions since we have not yet converged on the public
      # API for individual impurity density extensions.
      if attr_name == "impurity_fractions":
        continue

      attr_value = getattr(stacked_core_profiles, attr_name)

      output_key = output_name_map.get(attr_name, attr_name)

      # Skip _face attributes if their cell counterpart exists;
      # they are handled when the cell attribute is processed.
      if attr_name.endswith("_face") and (
          attr_name.removesuffix("_face") in core_profile_field_names
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
          data_to_save = _extend_cell_grid_to_boundaries(attr_value, face_value)
        xr_dict[output_key] = self._pack_into_data_array(
            output_key, data_to_save
        )
        continue

      if hasattr(attr_value, "cell_plus_boundaries"):
        # Handles stacked CellVariable-like objects.
        data_to_save = attr_value.cell_plus_boundaries()
      else:
        face_attr_name = f"{attr_name}_face"
        if face_attr_name in core_profile_field_names:
          # Combine cell and edge face values.
          face_value = getattr(stacked_core_profiles, face_attr_name)
          data_to_save = _extend_cell_grid_to_boundaries(attr_value, face_value)
        else:  # cell array with no face counterpart, or a scalar value
          data_to_save = attr_value

      xr_dict[output_key] = self._pack_into_data_array(output_key, data_to_save)

    # Handle derived quantities
    Ip_data = stacked_core_profiles.Ip_profile_face[..., -1]
    xr_dict[IP] = self._pack_into_data_array(IP, Ip_data)

    return xr_dict

  def _save_core_transport(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core transport to a dict."""
    xr_dict = {}
    core_transport = self._stacked_core_transport

    xr_dict[CHI_TURB_I] = core_transport.chi_face_ion
    xr_dict[CHI_TURB_E] = core_transport.chi_face_el
    xr_dict[D_TURB_E] = core_transport.d_face_el
    xr_dict[V_TURB_E] = core_transport.v_face_el

    xr_dict[CHI_NEO_I] = core_transport.chi_neo_i
    xr_dict[CHI_NEO_E] = core_transport.chi_neo_e
    xr_dict[D_NEO_E] = core_transport.D_neo_e
    xr_dict[V_NEO_E] = core_transport.V_neo_e
    xr_dict[V_NEO_WARE_E] = core_transport.V_neo_ware_e

    # Save optional BohmGyroBohm attributes if present.
    core_transport = self._stacked_core_transport
    if (
        core_transport.chi_face_el_bohm is not None
        or core_transport.chi_face_el_gyrobohm is not None
        or core_transport.chi_face_ion_bohm is not None
        or core_transport.chi_face_ion_gyrobohm is not None
    ):
      xr_dict[CHI_BOHM_E] = core_transport.chi_face_el_bohm
      xr_dict[CHI_GYROBOHM_E] = core_transport.chi_face_el_gyrobohm
      xr_dict[CHI_BOHM_I] = core_transport.chi_face_ion_bohm
      xr_dict[CHI_GYROBOHM_I] = core_transport.chi_face_ion_gyrobohm

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
            self._stacked_core_profiles.T_e.value
            - self._stacked_core_profiles.T_i.value
        )
    )

    xr_dict[J_BOOTSTRAP] = _extend_cell_grid_to_boundaries(
        self._stacked_core_sources.bootstrap_current.j_bootstrap,
        self._stacked_core_sources.bootstrap_current.j_bootstrap_face,
    )

    # Add source profiles with suffixes indicating which profile they affect.
    for profile in self._stacked_core_sources.T_i:
      if profile == "fusion":
        xr_dict["p_alpha_i"] = self._stacked_core_sources.T_i[profile]
      else:
        xr_dict[f"p_{profile}_i"] = self._stacked_core_sources.T_i[profile]
    for profile in self._stacked_core_sources.T_e:
      if profile == "fusion":
        xr_dict["p_alpha_e"] = self._stacked_core_sources.T_e[profile]
      else:
        xr_dict[f"p_{profile}_e"] = self._stacked_core_sources.T_e[profile]
    for profile in self._stacked_core_sources.psi:
      xr_dict[f"j_{profile}"] = self._stacked_core_sources.psi[profile]
    for profile in self._stacked_core_sources.n_e:
      xr_dict[f"s_{profile}"] = self._stacked_core_sources.n_e[profile]

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
    for field in dataclasses.fields(self._stacked_post_processed_outputs):
      attr_name = field.name

      # The impurity_radiation is structured differently and handled separately.
      if attr_name == "impurity_species":
        continue

      attr_value = getattr(self._stacked_post_processed_outputs, attr_name)
      if hasattr(attr_value, "cell_plus_boundaries"):
        # Handles stacked CellVariable-like objects.
        data_to_save = attr_value.cell_plus_boundaries()
      else:
        data_to_save = attr_value
      xr_dict[attr_name] = self._pack_into_data_array(attr_name, data_to_save)

    if self._stacked_post_processed_outputs.impurity_species:
      radiation_outputs = (
          impurity_radiation.construct_xarray_for_radiation_output(
              self._stacked_post_processed_outputs.impurity_species,
              self.times,
              self.rho_cell_norm,
              TIME,
              RHO_CELL_NORM,
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
          or field_name == "geometry_type"
          or field_name == "Ip_from_parameters"
          or field_name == "j_total"
          or not isinstance(data, array_typing.Array)
      ):
        continue
      if f"{field_name}_face" in geometry_attributes:
        data = _extend_cell_grid_to_boundaries(
            data, geometry_attributes[f"{field_name}_face"]
        )
      # Remap to avoid outputting _face suffix in output.
      if field_name.endswith("_face"):
        field_name = field_name.removesuffix("_face")
      if field_name == "Ip_profile":
        # Ip_profile exists in core profiles so rename to avoid duplicate.
        field_name = "Ip_profile_from_geo"
      if field_name == "psi":
        # Psi also exists in core profiles so rename to avoid duplicate.
        field_name = "psi_from_geo"
      if field_name == "_z_magnetic_axis":
        # This logic only reached if not None. Avoid leading underscore in name.
        field_name = "z_magnetic_axis"
      data_array = self._pack_into_data_array(
          field_name,
          data,
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
      if name in EXCLUDED_GEOMETRY_NAMES:
        continue
      if isinstance(value, property):
        property_data = value.fget(self._stacked_geometry)
        # Check if there is a corresponding face variable for this property.
        # If so, extend the data to the cell+boundaries grid.
        if f"{name}_face" in property_names:
          face_data = getattr(self._stacked_geometry, f"{name}_face")
          property_data = _extend_cell_grid_to_boundaries(
              property_data, face_data
          )
        data_array = self._pack_into_data_array(name, property_data)
        if data_array is not None:
          # Remap to avoid outputting _face suffix in output. Done only for
          # _face variables with no corresponding non-face variable.
          if name.endswith("_face"):
            name = name.removesuffix("_face")
          xr_dict[name] = data_array

    return xr_dict
