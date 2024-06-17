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

"""The Stepper class.

Abstract base class defining updates to State.
"""

import abc
import dataclasses

import jax
from torax import core_profile_setters
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import runtime_params as runtime_params_lib
from torax.transport_model import transport_model as transport_model_lib


class Stepper(abc.ABC):
  """Calculates a single time step's update to State.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    source_models: All TORAX sources used to compute both the explicit and
      implicit source profiles used for each time step as terms in the state
      evolution equations. Though the explicit profiles are computed outside the
      call to Stepper, the same sources should be used to compute those. The
      Sources are exposed here to provide a single source of truth for which
      sources are used during a run.
  """

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
  ):
    self.transport_model = transport_model
    self.source_models = source_models

  def __call__(
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
  ) -> tuple[
      state.CoreProfiles,
      source_profiles.SourceProfiles,
      state.CoreTransport,
      int,
  ]:
    """Applies a time step update.

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
        outside the possibly-JAX-jitted JointStateStepperCallable, they can be
        calculated in non-JAX-friendly ways.

    Returns:
      new_core_profiles: Updated core profiles.
      core_sources: Source profiles of the implicit sources, computed at the
        most recent guess for time t+dt. Any state-dependent source profiles
        will not be computed based on the exact state of the core profiles at
        time t+dt, but rather they will be computed based on the final guess the
        solver used while calculating coeffs in the solver.
      core_transport: Transport coefficients for time t+dt.
      error: 0 if step was successful (linear step, or nonlinear step with
        residual or loss under tolerance at exit), or 1 if unsuccessful,
        indicating that a rerun with a smaller timestep is needed
    """

    # This base class method can be completely overriden by a subclass, but
    # most can make use of the boilerplate here and just implement `_x_new`.

    # Use runtime params to determine which variables to evolve
    evolving_names = []
    if static_runtime_params_slice.ion_heat_eq:
      evolving_names.append('temp_ion')
    if static_runtime_params_slice.el_heat_eq:
      evolving_names.append('temp_el')
    if static_runtime_params_slice.current_eq:
      evolving_names.append('psi')
    if static_runtime_params_slice.dens_eq:
      evolving_names.append('ne')
    evolving_names = tuple(evolving_names)

    # Don't call solver functions on an empty list
    if evolving_names:
      x_new, core_sources, core_transport, error = self._x_new(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          evolving_names=evolving_names,
      )
    else:
      x_new = tuple()
      core_sources = source_models_lib.build_all_zero_profiles(
          source_models=self.source_models,
          geo=geo_t,
      )
      core_transport = state.CoreTransport.zeros(geo_t)
      error = 0

    core_profiles_t_plus_dt = (
        core_profile_setters.update_evolving_core_profiles(
            x_new,
            dynamic_runtime_params_slice_t_plus_dt,
            core_profiles_t_plus_dt,
            evolving_names,
        )
    )

    return (
        core_profiles_t_plus_dt,
        core_sources,
        core_transport,
        error,
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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      int,
  ]:
    """Calculates new values of the changing variables.

    Subclasses must either implement `_x_new` so that `Stepper.__call__`
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
      core_sources: see the docstring of __call__
      core_transport: Transport coefficients for time t+dt.
      error: 0 if step was successful (linear step, or nonlinear step with
        residual or loss under tolerance at exit), or 1 if unsuccessful,
        indicating that a rerun with a smaller timestep is needed
    """

    raise NotImplementedError(
        f'{type(self)} must implement `_x_new` or '
        'implement a different `__call__` that does not'
        ' need `_x_new`.'
    )


@dataclasses.dataclass(kw_only=True)
class StepperBuilder(abc.ABC):
  """Factory for Stepper objects."""

  @abc.abstractmethod
  def __call__(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
  ) -> Stepper:
    """Builds a Stepper instance."""

  # Input parameters to the stepper built by this class.
  runtime_params: runtime_params_lib.RuntimeParams = dataclasses.field(
      default_factory=runtime_params_lib.RuntimeParams
  )
