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
"""Conversion utilities between CoreProfiles state variables and fvm objects."""

import dataclasses
from typing import Tuple

from torax._src import state
from torax._src.fvm import cell_variable

# Can be extended to include other density variables when solver is generalized.
DENSITY_NAMES = 'n_e'


def core_profiles_to_solver_x_tuple(
    core_profiles: state.CoreProfiles,
    evolving_names: Tuple[str, ...],
) -> Tuple[cell_variable.CellVariable, ...]:
  """Converts evolving parts of CoreProfiles to the 'x' tuple for the solver.

  State variables in the solver are scaled for solver numerical conditioning.

  Args:
    core_profiles: The input CoreProfiles object.
    evolving_names: Tuple of strings naming the variables to be evolved by the
      solver.

  Returns:
    A tuple of CellVariable objects, one for each name in evolving_names,
    with density values appropriately scaled for the solver.
  """
  x_tuple_for_solver_list = []

  for name in evolving_names:
    cv_original = getattr(core_profiles, name)
    if name in DENSITY_NAMES:
      # Will be replaced in next step by constants.DENSITY_SCALING_FACTOR
      scaling_factor = 1.0
    else:
      scaling_factor = 1.0

    scaled_value = cv_original.value / scaling_factor

    scaled_right_face_constraint = None
    if cv_original.right_face_constraint is not None:
      scaled_right_face_constraint = (
          cv_original.right_face_constraint / scaling_factor
      )

    scaled_right_face_grad_constraint = None
    if cv_original.right_face_grad_constraint is not None:
      scaled_right_face_grad_constraint = (
          cv_original.right_face_grad_constraint / scaling_factor
      )

    solver_cv = dataclasses.replace(
        cv_original,
        value=scaled_value,
        right_face_constraint=scaled_right_face_constraint,
        right_face_grad_constraint=scaled_right_face_grad_constraint,
    )
    x_tuple_for_solver_list.append(solver_cv)

  return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(
    x_new: tuple[cell_variable.CellVariable, ...],
    evolving_names: tuple[str, ...],
    core_profiles: state.CoreProfiles,
) -> state.CoreProfiles:
  """Gets updated cell variables for evolving state variables in core_profiles.

  If a variable is in `evolving_names`, its new value is taken from `x_new`.
  Otherwise, the existing value from `core_profiles` is kept.
  State variables in the solver may be scaled for solver numerical conditioning,
  and must be scaled back to their original units before being written to
  `core_profiles`.

  Args:
    x_new: The new values of the evolving variables.
    evolving_names: The names of the evolving variables.
    core_profiles: The current set of core plasma profiles.

  Returns:
    An updated CoreProfiles object with the new values.
  """
  updated_vars = {}

  for var_name in evolving_names:
    updated_var = x_new[evolving_names.index(var_name)]
    if var_name in DENSITY_NAMES:
      # Will be made constants.DENSITY_SCALING_FACTOR in next step.
      scaling_factor = 1.0
      updated_var = dataclasses.replace(
          updated_var,
          value=updated_var.value * scaling_factor,
          right_face_constraint=updated_var.right_face_constraint
          * scaling_factor,
      )
    updated_vars[var_name] = updated_var
  return dataclasses.replace(core_profiles, **updated_vars)
