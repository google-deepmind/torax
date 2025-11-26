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

"""The Solver class.

Abstract base class defining updates to State.
"""

import abc
import dataclasses
import functools

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.sources import source_profiles


@dataclasses.dataclass(frozen=True, eq=False)
class Solver(static_dataclass.StaticDataclass, abc.ABC):
  """Solves for a single time step's update to State.


  Attributes:
    physics_models: Physics models.
  """

  physics_models: physics_models_lib.PhysicsModels

  @functools.partial(
      jax.jit,
      static_argnames=[
          'self',
      ],
  )
  def __call__(
      self,
      t: jax.Array,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Applies a time step update.

    Args:
      t: Time.
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step). These runtime params can change from step to step without
        triggering a recompilation.
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Geometry of the torus at time t.
      geo_t_plus_dt: Geometry of the torus at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: Source profiles of all explicit sources (as
        configured by the input params). All implicit source's profiles will be
        set to 0 in this object. These explicit source profiles were calculated
        either based on the original core profiles at the start of the time step
        or were independent of the core profiles. Because they were calculated
        outside the possibly-JAX-jitted solver logic, they can be calculated in
        non-JAX-friendly ways.

    Returns:
      x_new: Tuple containing new cell-grid values of the evolving variables.
      solver_numeric_output: Error and solver iteration info.
    """

    # This base class method can be completely overridden by a subclass, but
    # most can make use of the boilerplate here and just implement `_x_new`.

    # Don't call solver functions on an empty list
    if runtime_params_t.numerics.evolving_names:
      (
          x_new,
          solver_numeric_output,
      ) = self._x_new(
          dt=dt,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          evolving_names=runtime_params_t.numerics.evolving_names,
      )
    else:
      x_new = tuple()
      solver_numeric_output = state.SolverNumericOutputs(
          sawtooth_crash=False,
          solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
          inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      )

    return (
        x_new,
        solver_numeric_output,
    )

  def _x_new(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Calculates new values of the changing variables.

    Subclasses must either implement `_x_new` so that `Solver.__call__`
    will work, or implement a different `__call__`.

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step). These runtime params can change from step to step without
        triggering a recompilation.
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Geometry of the torus for time t.
      geo_t_plus_dt: Geometry of the torus for time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: see the docstring of __call__
      evolving_names: The names of core_profiles variables that should evolve.

    Returns:
      x_new: The values of the evolving variables at time t + dt.
      solver_numeric_output: Error and solver iteration info.
    """

    raise NotImplementedError(
        f'{type(self)} must implement `_x_new` or '
        'implement a different `__call__` that does not'
        ' need `_x_new`.'
    )
