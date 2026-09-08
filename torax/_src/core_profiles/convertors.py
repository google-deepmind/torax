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
"""Conversion, state scaling, and residual scaling utilities.

Provides utilities to convert between CoreProfiles state variables and FVM
solver objects, including state variable scaling and residual
scaling.

State variable scaling statically converts state variables from physical units
to dimensionless O(1) solver units, equilibrating step sizes across Jacobian
columns. Residual scaling dynamically weights residual equations by
characteristic physical profile scales, normalizing error measurements across
Jacobian rows for linesearch and convergence checks. These two scalings operate
orthogonally on the domain (unknowns) and codomain (equations) respectively.
"""

import dataclasses
from typing import Final, Mapping, Tuple

import immutabledict
import jax
import jax.numpy as jnp
from torax._src import array_typing
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


# Default physical floors for residual channel scaling (in solver units).
# Used when channel magnitudes are small or zero (e.g. cold start, zero axis
# flux).
RESIDUAL_SCALE_FLOORS: Final[Mapping[str, float]] = (
    immutabledict.immutabledict({
        'T_i': 0.1,  # 100 eV
        'T_e': 0.1,  # 100 eV
        'n_e': 0.01,  # 1e18 m^-3
        'psi': 0.01,  # 0.01 Wb
    })
)


def _compute_channel_residual_scale(
    name: str, x: array_typing.Array
) -> jax.Array:
  """Computes a dynamic characteristic physical scale for a channel.

  Args:
    name: The name of the physical channel (e.g., 'T_i', 'T_e', 'n_e', 'psi').
    x: The channel's profile array at the start of the time step (in solver
      units).

  Returns:
    A scalar jax.Array representing the characteristic physical scale.
  """
  floor = RESIDUAL_SCALE_FLOORS[name]
  if name == 'psi':
    return jnp.maximum(jnp.max(x) - jnp.min(x), floor)
  else:
    return jnp.maximum(jnp.mean(jnp.abs(x)), floor)


def compute_residual_scaling_vector(
    evolving_names: tuple[str, ...],
    x_old: tuple[cell_variable.CellVariable, ...],
) -> jax.Array:
  """Computes the residual scaling vector across all evolving channels and cells.

  Reference scales are adapted dynamically to the solution state rather than
  using hardcoded constants.

  For poloidal flux (psi), the scale is computed from the profile span
  (max - min), which is gauge-invariant. For temperatures and densities,
  the scale is computed from the mean profile magnitude. Physical floors
  prevent division by zero during cold startup or at zero crossings.

  Args:
    evolving_names: The names of the evolving variables.
    x_old: The tuple of CellVariable objects at the start of the time step (in
      solver units).

  Returns:
    A 1D jax.Array of scales corresponding to the flattened residual vector.
  """
  channel_scales = [
      jnp.full_like(cv.value, _compute_channel_residual_scale(name, cv.value))
      for name, cv in zip(evolving_names, x_old)
  ]
  return jnp.concatenate(channel_scales)
