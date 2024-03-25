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
from typing import Callable

import jax
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import geometry
from torax import state as state_module
from torax import update_state
from torax.sources import source_profiles
from torax.transport_model import transport_model as transport_model_lib


class Stepper(abc.ABC):
  """Calculates a single time step's update to State.

  Attributes:
    transport_model: A TransportModel subclass, calculates transport coeffs.
    sources: All TORAX sources used to compute both the explicit and implicit
      source profiles used for each time step as terms in the state evolution
      equations. Though the explicit profiles are computed outside the call to
      Stepper, the same sources should be used to compute those. The Sources are
      exposed here to provide a single source of truth for which sources are
      used during a run.
  """

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      sources: source_profiles.Sources,
  ):
    self.transport_model = transport_model
    self.sources = sources

  def __call__(
      self,
      sim_state_t: state_module.ToraxSimState,
      sim_state_t_plus_dt: state_module.ToraxSimState,
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      dt: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[state_module.State, int, calc_coeffs.AuxOutput]:
    """Applies a time step update.

    Args:
      sim_state_t: Full sim state at the beginning of the time step, including
        the "mesh_state" which includes the profiles that are evolved by the PDE
        system.
      sim_state_t_plus_dt: Full sim state which contains all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      geo: Geometry of the torus.
      dynamic_config_slice_t: Runtime configuration for time t (the start time
        of the step). These config params can change from step to step without
        triggering a recompilation.
      dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt,
        used for implicit calculations in the solver.
      static_config_slice: Input params that cannot change during the compiled
        lifetime of the joint state stepper, which wraps this stepper. These
        don't have to be JAX-friendly types and can be used in control-flow
        logic.
      dt: Time step duration.
      explicit_source_profiles: Source profiles of all explicit sources (as
        configured by the input config). All implicit source's profiles will be
        set to 0 in this object. These explicit source profiles were calculated
        either based on the original state at the start of the time step or were
        independent of the state. Because they were calculated outside the
        possibly-JAX-jitted JointStateStepperCallable, they can be calculated in
        non-JAX-friendly ways.

    Returns:
      new_state: Updated sim state.
      error: 0 if step was successful (linear step, or nonlinear step with
        residual or loss under tolerance at exit), or 1 if unsuccessful,
        indicating that a rerun with a smaller timestep is needed
      aux_output: Extra outputs useful to inspect other values while the
        coeffs are computed.
    """

    # This base class method can be completely overriden by a subclass, but
    # most can make use of the boilerplate here and just implement `_x_new`.

    # Use config to determine which variables to evolve
    evolving_names = []
    if static_config_slice.ion_heat_eq:
      evolving_names.append('temp_ion')
    if static_config_slice.el_heat_eq:
      evolving_names.append('temp_el')
    if static_config_slice.current_eq:
      evolving_names.append('psi')
    if static_config_slice.dens_eq:
      evolving_names.append('ne')
    evolving_names = tuple(evolving_names)

    # Don't call solver functions on an empty list
    if evolving_names:
      x_new, error, aux_output = self._x_new(
          sim_state_t=sim_state_t,
          sim_state_t_plus_dt=sim_state_t_plus_dt,
          evolving_names=evolving_names,
          geo=geo,
          dynamic_config_slice_t=dynamic_config_slice_t,
          dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
          static_config_slice=static_config_slice,
          dt=dt,
          explicit_source_profiles=explicit_source_profiles,
      )
    else:
      x_new = tuple()
      error = 0
      aux_output = calc_coeffs.AuxOutput.build_from_geo(geo)

    state_t_plus_dt = update_state.update_state(
        sim_state_t_plus_dt.mesh_state,
        x_new,
        evolving_names,
        dynamic_config_slice_t_plus_dt,
    )

    return (
        state_t_plus_dt,
        error,
        aux_output,
    )

  def _x_new(
      self,
      sim_state_t: state_module.ToraxSimState,
      sim_state_t_plus_dt: state_module.ToraxSimState,
      evolving_names: tuple[str, ...],
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      dt: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """Calculates new values of the changing variables.

    Subclasses must either implement `_x_new` so that `Stepper.__call__`
    will work, or implement a different `__call__`.

    Args:
      sim_state_t: The full simulation state at time t.
      sim_state_t_plus_dt: Full sim state which contains all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      evolving_names: The names of state variables that should evolve.
      geo: Geometry of the torus.
      dynamic_config_slice_t: Runtime configuration for time t (the start time
        of the step). These config params can change from step to step without
        triggering a recompilation.
      dynamic_config_slice_t_plus_dt: Runtime configuration for time t + dt,
        used for implicit calculations in the solver.
      static_config_slice: Input params that cannot change during the compiled
        lifetime of the joint state stepper, which wraps this stepper. These
        don't have to be JAX-friendly types and can be used in control-flow
        logic.
      dt: Time step duration.
      explicit_source_profiles: see the docstring of __call__

    Returns:
      x_new: The values of the evolving variables at time t + dt.
      error: 0 if step was successful, 1 if residual or loss under tolerance
      aux_output: Extra outputs useful to inspect other values while the
        coeffs are computed.
    """

    raise NotImplementedError(
        f'{type(self)} must implement `_x_new` or '
        'implement a different `__call__` that does not'
        ' need `_x_new`.'
    )


StepperBuilder = Callable[
    [  # Arguments
        transport_model_lib.TransportModel,
        source_profiles.Sources,
    ],
    Stepper,  # Returns a Stepper.
]
