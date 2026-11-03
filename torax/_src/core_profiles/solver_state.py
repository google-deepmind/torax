# Copyright 2026 DeepMind Technologies Limited
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

"""Primary profiles that can be prescribed or evolved by the PDE solver."""

import dataclasses
import jax
from torax._src import array_typing
from torax._src.fvm import cell_variable

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SolverState:
  """Primary profiles and step-level boundary conditions for the PDE solver.

  For a candidate step at t + dt, boundary conditions and prescribed profile
  values are populated across [t, t + dt] prior to the solver loop (in
  `updaters.provide_solver_state_t_plus_dt`), and iterative solvers replace
  only the cell values of channels in `evolving_names`.

  Attributes:
    T_i: Ion temperature [keV].
    T_e: Electron temperature [keV].
    psi: Poloidal flux [Wb].
    n_e: Electron density [m^-3].
    toroidal_angular_velocity: Toroidal angular velocity [rad/s].
    psidot_right_face_grad_constraint: Right-face radial gradient constraint on
      psidot [Wb/s], equal to the finite-difference rate of change of
      `psi.right_face_grad_constraint` over `[t, t + dt]`. Stored on
      `SolverState` because it is a two-time-level step boundary condition
      needed by single-time-slice `evaluate_state`, whereas `psidot`'s
      `CellVariable` is not constructed until `CoreProfiles`. None under
      `v_loop_lcfs` boundary conditions.
  """

  T_i: cell_variable.CellVariable
  T_e: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  toroidal_angular_velocity: cell_variable.CellVariable
  psidot_right_face_grad_constraint: array_typing.FloatScalar | None = None
