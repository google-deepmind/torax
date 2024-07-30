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

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
import xarray as xr


class StateHistory:
  """A history of the state of the simulation."""

  def __init__(self, states: tuple[state.ToraxSimState, ...]):
    core_profiles = [state.core_profiles.history_elem() for state in states]
    core_sources = [state.core_sources for state in states]
    transport = [state.core_transport for state in states]
    stack = lambda *ys: jnp.stack(ys)
    self.core_profiles = jax.tree_util.tree_map(stack, *core_profiles)
    self.core_sources = jax.tree_util.tree_map(stack, *core_sources)
    self.core_transport = jax.tree_util.tree_map(stack, *transport)
    self.times = jnp.array([state.t for state in states])
    chex.assert_rank(self.times, 1)

  def simulation_output_to_xr(
      self,
      geo: geometry.Geometry,
  ) -> xr.Dataset:
    """Build an xr.Dataset of the simulation output."""
    # TODO(b/338033916) Document what is being output.

    # TODO(b/338033916). Extend outputs with:
    # Post-processed integrals, more geo outputs.
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    #  Exclude uninteresting variables from output DataSet
    exclude_set = {
        "explicit_e",
        "explicit_i",
        "implicit_ee",
        "implicit_ii",
        "implicit_ei",
        "implicit_ie",
        "qei_coef",
        "sigma_face",
    }

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

    # Build a PyTree of variables we will want to log.
    tree = (
        self.core_profiles,
        self.core_transport,
        self.core_sources,
    )

    # Only try to log arrays.
    leaves_with_path = jax.tree_util.tree_leaves_with_path(
        tree, is_leaf=lambda x: isinstance(x, jax.Array)
    )

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

    # Extend with desired core_profiles, core_sources, core_transport variables
    for path, leaf in leaves_with_path:
      name, da = _translate_leaf_with_path(time, geo, path, leaf)
      if da is not None and name not in exclude_set:
        xr_dict[name] = da
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


def _translate_leaf_with_path(
    time: xr.DataArray,
    geo: geometry.Geometry,
    path: tuple[Any, ...],
    leaf: jax.Array,
) -> tuple[str, xr.DataArray | None]:
  """Logic for converting and filtering which paths we want to save."""
  # Functions to check if a leaf is a face or cell variable
  # Assume that all arrays with shape (time, rho_face) are face variables
  # and all arrays with shape (time, rho_cell) are cell variables
  is_face_var = lambda x: x.ndim == 2 and x.shape == (
      len(time),
      len(geo.r_face),
  )
  is_cell_var = lambda x: x.ndim == 2 and x.shape == (len(time), len(geo.r))
  is_scalar = lambda x: x.ndim == 1 and x.shape == (len(time),)

  name = path_to_name(path)

  if is_face_var(leaf):
    return name, xr.DataArray(leaf, dims=["time", "rho_face"], name=name)
  elif is_cell_var(leaf):
    return name, xr.DataArray(leaf, dims=["time", "rho_cell"], name=name)
  elif is_scalar(leaf):
    return name, xr.DataArray(leaf, dims=["time"], name=name)
  else:
    return name, None


# TODO(b/338033916): Modify to specify exactly which outputs we would like.
def path_to_name(
    path: tuple[Any, ...],
) -> str:
  """Converts paths to names of variables."""
  # Aliases for the names of the variables.
  name_map = {
      "fusion_heat_source_el": "Qfus_e",
      "fusion_heat_source_ion": "Qfus_i",
      "generic_ion_el_heat_source_el": "Qext_e",
      "generic_ion_el_heat_source_ion": "Qext_i",
      "ohmic_heat_source": "Qohm",
      "qei_source": "Qei",
      "gas_puff_source": "s_puff",
      "nbi_particle_source": "s_nbi",
      "pellet_source": "s_pellet",
      "bremsstrahlung_heat_sink": "Qbrem",
  }
  initial_cond_vars = [
      "temp_ion",
      "temp_el",
      "ni",
      "ne",
  ]

  if isinstance(path[-1], jax.tree_util.DictKey):
    name = path[-1].key
  elif path[-1].name == "value":
    name = path[-2].name
  # For the initial conditions we want to save the right face constraint.
  elif (
      isinstance(path[-2], jax.tree_util.GetAttrKey)
      and path[-2].name in initial_cond_vars
      and path[-1].name == "right_face_constraint"
  ):
    name = path[-2].name + "_right_bc"
  # For the psi variable we want to save the value and the right grad face
  # constraints as well.
  elif (
      isinstance(path[-2], jax.tree_util.GetAttrKey)
      and path[-2].name == "psi"
      and path[-1].name == "right_face_grad_constraint"
  ):
    name = path[-2].name + "_right_grad_bc"
  else:
    name = path[-1].name

  if name in name_map:
    return name_map[name]

  return name
