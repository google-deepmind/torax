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

from absl import logging
import chex
import jax
import numpy as np
from torax._src import constants
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import model_config
import xarray as xr

import os


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

# Currents.
J_TOTAL = "j_total"
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"
IP_PROFILE = "Ip_profile"
SIGMA_PARALLEL = "sigma_parallel"

IP = "Ip"
V_LOOP_LCFS = "v_loop_lcfs"

# Core transport.
CHI_TURB_I = "chi_turb_i"
CHI_TURB_E = "chi_turb_e"
D_TURB_E = "D_turb_e"
V_TURB_E = "V_turb_e"
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
    cell_var: chex.Array, face_var: chex.Array
) -> chex.Array:
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
    # Rescale output CoreProfiles to have density units in m^-3.
    # This is done to maintain the same external API following an upcoming
    # change to the internal CoreProfiles density units.
    self._stacked_core_profiles = _rescale_core_profiles(
        self._stacked_core_profiles
    )
    self._stacked_core_sources: source_profiles_lib.SourceProfiles = (
        jax.tree_util.tree_map(stack, *self._core_sources)
    )
    self._stacked_core_transport: state.CoreTransport = jax.tree_util.tree_map(
        stack, *self._transport
    )
    self._stacked_post_processed_outputs: (
        post_processing.PostProcessedOutputs
    ) = jax.tree_util.tree_map(
        stack, *post_processed_outputs_history
    )
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
  def times(self) -> chex.Array:
    """Returns the time of the simulation."""
    return self._times

  @property
  def rho_cell_norm(self) -> chex.Array:
    """Returns the normalized toroidal coordinate on the cell grid."""
    return self._rho_cell_norm

  @property
  def rho_face_norm(self) -> chex.Array:
    """Returns the normalized toroidal coordinate on the face grid."""
    return self._rho_face_norm

  @property
  def rho_norm(self) -> chex.Array:
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

  def simulation_output_to_xr(
      self,
      file_restart: file_restart_pydantic_model.FileRestart | None = None,
  ) -> xr.DataTree:
    """Build an xr.DataTree of the simulation output.

    Args:
      file_restart: If provided, contains information on a file this sim was
        restarted from, this is useful in case we want to stitch that to the
        beginning of this sim output.

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
        if v is not None and v.values.ndim == 2  # pytype: disable=attribute-error
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

    if file_restart is not None and file_restart.stitch:
      data_tree = stitch_state_files(file_restart, data_tree)

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
    """Saves the core profiles to a dict."""
    xr_dict = {}
    core_profiles = self._stacked_core_profiles

    xr_dict[T_E] = core_profiles.T_e.cell_plus_boundaries()
    xr_dict[T_I] = core_profiles.T_i.cell_plus_boundaries()
    xr_dict[PSI] = core_profiles.psi.cell_plus_boundaries()
    xr_dict[V_LOOP] = core_profiles.psidot.cell_plus_boundaries()
    xr_dict[N_E] = core_profiles.n_e.cell_plus_boundaries()
    xr_dict[N_I] = core_profiles.n_i.cell_plus_boundaries()
    xr_dict[N_IMPURITY] = core_profiles.n_impurity.cell_plus_boundaries()
    xr_dict[Z_IMPURITY] = _extend_cell_grid_to_boundaries(
        core_profiles.Z_impurity, core_profiles.Z_impurity_face
    )
    xr_dict[SIGMA_PARALLEL] = core_profiles.sigma

    # Currents.
    xr_dict[J_TOTAL] = _extend_cell_grid_to_boundaries(
        core_profiles.j_total, core_profiles.j_total_face
    )
    xr_dict[IP_PROFILE] = core_profiles.Ip_profile_face
    xr_dict[IP] = core_profiles.Ip_profile_face[:, -1]

    xr_dict[Q] = core_profiles.q_face
    xr_dict[MAGNETIC_SHEAR] = core_profiles.s_face

    xr_dict[V_LOOP_LCFS] = core_profiles.v_loop_lcfs

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

    xr_dict[CHI_TURB_I] = self._stacked_core_transport.chi_face_ion
    xr_dict[CHI_TURB_E] = self._stacked_core_transport.chi_face_el
    xr_dict[D_TURB_E] = self._stacked_core_transport.d_face_el
    xr_dict[V_TURB_E] = self._stacked_core_transport.v_face_el

    # Save optional BohmGyroBohm attributes if nonzero.
    core_transport = self._stacked_core_transport
    if (
        np.any(core_transport.chi_face_el_bohm != 0)
        or np.any(core_transport.chi_face_el_gyrobohm != 0)
        or np.any(core_transport.chi_face_ion_bohm != 0)
        or np.any(core_transport.chi_face_ion_gyrobohm != 0)
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
    for field_name, data in dataclasses.asdict(
        self._stacked_post_processed_outputs
    ).items():
      xr_dict[field_name] = self._pack_into_data_array(field_name, data)
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
          or "face" in field_name
          or field_name == "geometry_type"
          or field_name == "Ip_from_parameters"
          or field_name == "j_total"
          or not isinstance(data, jax.Array)
      ):
        continue
      if f"{field_name}_face" in geometry_attributes:
        data = _extend_cell_grid_to_boundaries(
            data, geometry_attributes[f"{field_name}_face"]
        )
      if field_name == "psi":
        # Psi also exists in core profiles so rename to avoid duplicate.
        field_name = "psi_from_geo"
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
      if "face" in name:
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
          xr_dict[name] = data_array

    # Remap to avoid outputting _face suffix in output.
    g0_over_vpr_data_array = self._pack_into_data_array(
        "g0_over_vpr", self._stacked_geometry.g0_over_vpr_face
    )
    if g0_over_vpr_data_array is not None:
      xr_dict["g0_over_vpr"] = g0_over_vpr_data_array

    return xr_dict


def _rescale_core_profiles(
    core_profiles: state.CoreProfiles,
) -> state.CoreProfiles:
  """Rescale core profiles densities to be in m^-3."""
  return dataclasses.replace(
      core_profiles,
      n_e=dataclasses.replace(
          core_profiles.n_e,
          value=core_profiles.n_e.value * constants.DENSITY_SCALING_FACTOR,
          right_face_constraint=core_profiles.n_e.right_face_constraint
          * constants.DENSITY_SCALING_FACTOR,
      ),
      n_i=dataclasses.replace(
          core_profiles.n_i,
          value=core_profiles.n_i.value * constants.DENSITY_SCALING_FACTOR,
          right_face_constraint=core_profiles.n_i.right_face_constraint
          * constants.DENSITY_SCALING_FACTOR,
      ),
      n_impurity=dataclasses.replace(
          core_profiles.n_impurity,
          value=core_profiles.n_impurity.value
          * constants.DENSITY_SCALING_FACTOR,
          right_face_constraint=core_profiles.n_impurity.right_face_constraint
          * constants.DENSITY_SCALING_FACTOR,
      ),
  )
