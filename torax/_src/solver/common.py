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

"""Common solver logic."""

import equinox as eqx
import jax


class SolverError(eqx.Enumeration):
  """Solver error states enum."""

  converged = 'solver successfully converged within tolerance.'
  not_converged = 'solver failed to converge within tolerance.'
  weakly_converged = (
      'solver did not strictly converge but is still within reasonable'
      ' tolerance (`coarse_tol`)'
  )


def get_solver_error(
    scalar_condition: jax.Array,
    tol: float,
    coarse_tol: float | None = None,
) -> SolverError:
  """Returns the solver error state based on the scalar condition."""

  if scalar_condition.ndim != 0:
    raise ValueError(
        'scalar_condition must be a scalar, but has shape'
        f' {scalar_condition.shape}'
    )
  if coarse_tol is None:
    return jax.lax.cond(
        scalar_condition < tol,
        lambda: SolverError.converged,
        lambda: SolverError.not_converged,
    )
  else:
    return jax.lax.cond(
        scalar_condition < tol,
        lambda: SolverError.converged,
        lambda: jax.lax.cond(
            scalar_condition < coarse_tol,
            lambda: SolverError.weakly_converged,
            lambda: SolverError.not_converged,
        ),
    )
