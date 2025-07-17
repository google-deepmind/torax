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
import functools

import jax
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import xnp
from torax._src.config import runtime_params_slice
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.sources import source_profiles
import typing_extensions


class Solver(abc.ABC):
  """Solves for a single time steps update to State.

  Attributes:
    static_runtime_params_slice: Static runtime parameters. Input params that
      trigger recompilation when they change. These don't have to be
      JAX-friendly types and can be used in control-flow logic.
    physics_models: Physics models.
  """

  def __init__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      physics_models: physics_models_lib.PhysicsModels,
  ):
    self.static_runtime_params_slice = static_runtime_params_slice
    self.physics_models = physics_models

  def __hash__(self) -> int:
    return hash((
        self.static_runtime_params_slice,
        self.physics_models,
    ))

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return (
        self.static_runtime_params_slice == other.static_runtime_params_slice
        and self.static_runtime_params_slice
        == other.static_runtime_params_slice
        and self.physics_models == other.physics_models
    )

  @functools.cached_property
  def evolving_names(self) -> tuple[str, ...]:
    """The names of core_profiles variables that are evolved by the solver."""
    evolving_names = []
    if self.static_runtime_params_slice.numerics.evolve_ion_heat:
      evolving_names.append('T_i')
    if self.static_runtime_params_slice.numerics.evolve_electron_heat:
      evolving_names.append('T_e')
    if self.static_runtime_params_slice.numerics.evolve_current:
      evolving_names.append('psi')
    if self.static_runtime_params_slice.numerics.evolve_density:
      evolving_names.append('n_e')
    return tuple(evolving_names)

  @functools.partial(
      xnp.jit,
      static_argnames=[
          'self',
      ],
  )
  def __call__(
      self,
      t: jax.Array,
      dt: jax.Array,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
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
      dynamic_runtime_params_slice_t: Runtime parameters for time t (the start
        time of the step). These runtime params can change from step to step
        without triggering a recompilation.
      dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t +
        dt, used for implicit calculations in the solver.
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
    if self.evolving_names:
      (
          x_new,
          solver_numeric_output,
      ) = self._x_new(
          dt=dt,
          static_runtime_params_slice=self.static_runtime_params_slice,
          dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          evolving_names=self.evolving_names,
      )
    else:
      x_new = tuple()
      solver_numeric_output = state.SolverNumericOutputs()

    return (
        x_new,
        solver_numeric_output,
    )

  def _x_new(
      self,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
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
      static_runtime_params_slice: Input params that trigger recompilation when
        they change. These don't have to be JAX-friendly types and can be used
        in control-flow logic.
      dynamic_runtime_params_slice_t: Runtime parameters for time t (the start
        time of the step). These runtime params can change from step to step
        without triggering a recompilation.
      dynamic_runtime_params_slice_t_plus_dt: Runtime parameters for time t +
        dt, used for implicit calculations in the solver.
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
