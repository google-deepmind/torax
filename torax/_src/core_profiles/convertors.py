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
"""Conversion and scaling utilities between CoreProfiles and fvm objects."""

import dataclasses
from typing import Final, Mapping, Tuple

import immutabledict
from torax._src import state
from torax._src.fvm import cell_variable

# Factors for state variable scaling.
# Converts physical SI units -> dimensionless solver state units
# (x_solver = x_phys / S).
# Corresponds to column scaling of the Jacobian matrix.
STATE_SCALING_FACTORS: Final[Mapping[str, float]] = (
    immutabledict.immutabledict({
        'T_i': 1.0,
        'T_e': 1.0,
        'n_e': 1e20,
        'psi': 1.0,
    })
)


def core_profiles_to_solver_x_tuple(
    core_profiles: state.CoreProfiles,
    evolving_names: Tuple[str, ...],
) -> Tuple[cell_variable.CellVariable, ...]:
  """Converts evolving parts of CoreProfiles to the 'x' tuple for the solver.

  Applies state variable scaling to the solution state vector elements
  for numerical conditioning of the solver. This ensures that the state vector
  components are of similar order of magnitude (O(1)), corresponding to column
  equilibration of the Jacobian matrix.

  Args:
    core_profiles: The input CoreProfiles object.
    evolving_names: Tuple of strings naming the variables to be evolved by the
      solver.

  Returns:
    A tuple of CellVariable objects, one for each name in evolving_names,
    with values state-scaled for the solver.
  """
  x_tuple_for_solver_list = []

  for name in evolving_names:
    original_units_cv = getattr(core_profiles, name)
    solver_x_tuple_cv = apply_state_scaling(
        cv=original_units_cv,
        scaling_factor=1 / STATE_SCALING_FACTORS[name],
    )
    x_tuple_for_solver_list.append(solver_x_tuple_cv)

  return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(
    x_new: tuple[cell_variable.CellVariable, ...],
    evolving_names: tuple[str, ...],
    core_profiles: state.CoreProfiles,
) -> state.CoreProfiles:
  """Gets updated cell variables for evolving state variables in core_profiles.

  If a variable is in `evolving_names`, its new value is taken from `x_new`.
  Otherwise, the existing value from `core_profiles` is kept.
  State variables in the solver are state-scaled for solver numerical
  conditioning, and must be unscaled back to their original physical units
  before being written to `core_profiles`.

  Args:
    x_new: The state-scaled values of the evolving variables from the solver.
    evolving_names: The names of the evolving variables.
    core_profiles: The current set of core plasma profiles.

  Returns:
    An updated CoreProfiles object with physical-unit values.
  """
  updated_vars = {}

  for i, var_name in enumerate(evolving_names):
    solver_x_tuple_cv = x_new[i]
    # Unscale state scaling from solver (multiply by state scaling factor)
    original_units_cv = apply_state_scaling(
        cv=solver_x_tuple_cv,
        scaling_factor=STATE_SCALING_FACTORS[var_name],
    )
    updated_vars[var_name] = original_units_cv

  return dataclasses.replace(core_profiles, **updated_vars)


def apply_state_scaling(
    cv: cell_variable.CellVariable,
    scaling_factor: float,
) -> cell_variable.CellVariable:
  """Scales or unscales a CellVariable's relevant fields.

  Args:
    cv: The CellVariable to scale.
    scaling_factor: The factor to scale values and boundary conditions by.

  Returns:
    A new CellVariable with scaled or unscaled values.
  """
  operation = lambda x, factor: x * factor if x is not None else None

  scaled_value = operation(cv.value, scaling_factor)

  # Only scale constraints if they are not None
  scaled_left_face_constraint = operation(
      cv.left_face_constraint, scaling_factor
  )
  scaled_left_face_grad_constraint = operation(
      cv.left_face_grad_constraint, scaling_factor
  )
  scaled_right_face_constraint = operation(
      cv.right_face_constraint, scaling_factor
  )
  scaled_right_face_grad_constraint = operation(
      cv.right_face_grad_constraint, scaling_factor
  )

  return cell_variable.CellVariable(
      value=scaled_value,  # pyrefly: ignore[bad-argument-type]
      face_centers=cv.face_centers,
      left_face_constraint=scaled_left_face_constraint,
      left_face_grad_constraint=scaled_left_face_grad_constraint,
      right_face_constraint=scaled_right_face_constraint,
      right_face_grad_constraint=scaled_right_face_grad_constraint,
  )
