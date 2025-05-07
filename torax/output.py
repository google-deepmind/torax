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

import dataclasses
import inspect
import itertools

from absl import logging
import chex
import jax
import numpy as np
from torax import state
from torax.geometry import geometry as geometry_lib
from torax.sources import qei_source as qei_source_lib
from torax.sources import source_profiles
from torax.torax_pydantic import file_restart as file_restart_pydantic_model
from torax.torax_pydantic import model_config
import xarray as xr

import os


# Dataset names.
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"

# Core profiles.
TEMPERATURE_ELECTRON = "T_e"
TEMPERATURE_ION = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
MAGNETIC_SHEAR = "magnetic_shear"
N_IMPURITY = "n_impurity"
N_REF = "n_ref"
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
VLOOP_LCFS = "vloop_lcfs"

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
      state_history: tuple[state.ToraxSimState, ...],
      post_processed_outputs_history: tuple[state.PostProcessedOutputs, ...],
      sim_error: state.SimError,
      torax_config: model_config.ToraxConfig,
  ):
    self.sim_error = sim_error
    self.torax_config = torax_config
    core_profiles = [state.core_profiles for state in state_history]
    core_sources = [state.core_sources for state in state_history]
    transport = [state.core_transport for state in state_history]
    geometries = [state.geometry for state in state_history]
    self.geometry = geometry_lib.stack_geometries(geometries)
    stack = lambda *ys: np.stack(ys)
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
        jax.tree_util.tree_map(stack, *post_processed_outputs_history)
    )
    self.times = np.array([state.t for state in state_history])
    # The rho grid does not change in time so we can just take the first one.
    self.rho_cell_norm = state_history[0].geometry.rho_norm
    self.rho_face_norm = state_history[0].geometry.rho_face_norm
    self.rho_norm = np.concatenate(
        [[0.0], self.rho_cell_norm, [1.0]]
    )
    chex.assert_rank(self.times, 1)

    self.sawtooth_crash = np.array(
        [state.sawtooth_crash for state in state_history]
    )

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
    core_profiles = self.core_profiles

    xr_dict[TEMPERATURE_ELECTRON] = core_profiles.temp_el.cell_plus_boundaries()
    xr_dict[TEMPERATURE_ION] = core_profiles.temp_ion.cell_plus_boundaries()
    xr_dict[PSI] = core_profiles.psi.cell_plus_boundaries()
    xr_dict[V_LOOP] = core_profiles.psidot.cell_plus_boundaries()
    xr_dict[N_E] = core_profiles.n_e.cell_plus_boundaries()
    xr_dict[N_I] = core_profiles.ni.cell_plus_boundaries()
    xr_dict[N_IMPURITY] = core_profiles.nimp.cell_plus_boundaries()
    xr_dict[Z_IMPURITY] = _extend_cell_grid_to_boundaries(
        core_profiles.Zimp, core_profiles.Zimp_face
    )

    # Currents.
    xr_dict[J_TOTAL] = _extend_cell_grid_to_boundaries(
        core_profiles.currents.jtot, core_profiles.currents.jtot_face
    )
    xr_dict[J_OHMIC] = core_profiles.currents.johm
    xr_dict[J_EXTERNAL] = core_profiles.currents.external_current_source
    xr_dict[J_BOOTSTRAP] = _extend_cell_grid_to_boundaries(
        core_profiles.currents.j_bootstrap,
        core_profiles.currents.j_bootstrap_face,
    )
    xr_dict[IP_PROFILE] = core_profiles.currents.Ip_profile_face
    xr_dict[IP] = core_profiles.currents.Ip_total
    xr_dict[I_BOOTSTRAP] = core_profiles.currents.I_bootstrap
    xr_dict[SIGMA_PARALLEL] = core_profiles.currents.sigma

    xr_dict[Q] = core_profiles.q_face
    xr_dict[MAGNETIC_SHEAR] = core_profiles.s_face
    xr_dict[N_REF] = core_profiles.density_reference

    xr_dict[VLOOP_LCFS] = core_profiles.vloop_lcfs

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

    xr_dict[CHI_TURB_I] = self.core_transport.chi_face_ion
    xr_dict[CHI_TURB_E] = self.core_transport.chi_face_el
    xr_dict[D_TURB_E] = self.core_transport.d_face_el
    xr_dict[V_TURB_E] = self.core_transport.v_face_el

    # Save optional BohmGyroBohm attributes if nonzero.
    core_transport = self.core_transport
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
        self.core_sources.qei.qei_coef
        * (self.core_profiles.temp_el.value - self.core_profiles.temp_ion.value)
    )

    # Add source profiles with suffixes indicating which profile they affect.
    for profile in self.core_sources.temp_ion:
      if profile == "fusion":
        xr_dict["p_alpha_i"] = self.core_sources.temp_ion[profile]
      else:
        xr_dict[f"p_{profile}_i"] = self.core_sources.temp_ion[profile]
    for profile in self.core_sources.temp_el:
      if profile == "fusion":
        xr_dict["p_alpha_e"] = self.core_sources.temp_el[profile]
      else:
        xr_dict[f"p_{profile}_e"] = self.core_sources.temp_el[profile]
    for profile in self.core_sources.psi:
      xr_dict[f"j_{profile}"] = self.core_sources.psi[profile]
    for profile in self.core_sources.n_e:
      xr_dict[f"s_{profile}"] = self.core_sources.n_e[profile]

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
    geometry_attributes = dataclasses.asdict(self.geometry)

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
      file_restart: file_restart_pydantic_model.FileRestart | None = None,
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
        - rho_norm: The normalized toroidal coordinate on the cell grid with the
            left and right face boundaries added.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - sawtooth_crash: Time-series boolean indicating whether the
            state corresponds to a post-sawtooth-crash state.
        - sim_error: The simulation error state.
        - config: The ToraxConfig used to run the simulation serialized to JSON.
      The child datasets contain the following variables:
        - core_profiles: Contains data variables for quantities in the
          CoreProfiles.
        - core_sources: Contains data variables for quantities in the
          CoreSources.
        - post_processed_outputs: Contains data variables for quantities in the
          PostProcessedOutputs.
        - geometry: Contains data variables for quantities in the Geometry.
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
            self.sawtooth_crash, dims=[TIME], name=SAWTOOTH_CRASH
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
