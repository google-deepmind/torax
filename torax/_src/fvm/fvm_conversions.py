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
"""Conversions utilities for fvm objects."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import state
from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable


def cell_variable_tuple_to_vec(
    x_tuple: tuple[cell_variable.CellVariable, ...],
) -> jax.Array:
  """Converts a tuple of CellVariables to a flat array.

  Args:
    x_tuple: A tuple of CellVariables.

  Returns:
    A flat array of evolving state variables.
  """
  x_vec = jnp.concatenate([x.value for x in x_tuple])
  return x_vec


def vec_to_cell_variable_tuple(
    x_vec: jax.Array,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> tuple[cell_variable.CellVariable, ...]:
  """Converts a flat array of core profile state vars to CellVariable tuple.

  Requires an input CoreProfiles to provide boundary condition information.

  Args:
    x_vec: A flat array of evolving core profile state variables. The order of
      the variables in the array must match the order of the evolving_names.
    core_profiles: CoreProfiles containing all CellVariables with appropriate
      boundary conditions.
    evolving_names: The names of the evolving cell variables.

  Returns:
    A tuple of updated CellVariables.
  """
  x_split = jnp.split(x_vec, len(evolving_names))

  # First scale the core profiles to match the scaling in x_split, then
  # update the values in the scaled core profiles with new values from x_split.
  scaled_evolving_cp_list = [
      convertors.scale_cell_variable(
          getattr(core_profiles, name),
          scaling_factor=1 / convertors.SCALING_FACTORS[name],
      )
      for name in evolving_names
  ]

  x_out = [
      dataclasses.replace(
          scaled_evolving_cp,
          value=value,
      )
      for scaled_evolving_cp, value in zip(scaled_evolving_cp_list, x_split)
  ]

  return tuple(x_out)
