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

"""Functionality for running simulations.

This includes the `run_simulation` main loop, logging functionality,
and functionality for translating between our particular physics
simulation and generic fluid dynamics PDE solvers.

Use the TORAX_COMPILATION_ENABLED environment variable to turn
jax compilation off and on. Compilation is on by default. Turning
compilation off can sometimes help with debugging (e.g. by making
it easier to print error messages in context).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional, Protocol

from absl import logging
import jax
import jax.numpy as jnp
from torax import calc_coeffs
from torax import core_profile_setters
from torax import fvm
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.spectators import spectator as spectator_lib
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import time_step_calculator as ts
from torax.transport_model import transport_model as transport_model_lib


def _log_timestep(t: jax.Array, dt: jax.Array, stepper_iterations: int) -> None:
  """Logs basic timestep info."""
  logging.info(
      '\nSimulation time: %.5f, previous dt: %.6f, previous stepper'
      ' iterations: %d',
      t,
      dt,
      stepper_iterations,
  )


class CoeffsCallback:
  """Implements fvm.Block1DCoeffsCallback using calc_coeffs.

  Attributes:
    static_runtime_params_slice: See the docstring for `stepper.Stepper`.
    geo: See the docstring for `stepper.Stepper`.
    core_profiles_t: The core plasma profiles at the start of the time step.
    core_profiles_t_plus_dt: Core plasma profiles at the end of the time step.
    transport_model: See the docstring for `stepper.Stepper`.
    explicit_source_profiles: See the docstring for `stepper.Stepper`.
    source_models: See the docstring for `stepper.Stepper`.
    evolving_names: The names of the evolving variables.
  """

  def __init__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      transport_model: transport_model_lib.TransportModel,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      source_models: source_models_lib.SourceModels,
      evolving_names: tuple[str, ...],
  ):
    self.static_runtime_params_slice = static_runtime_params_slice
    self.geo = geo
    self.core_profiles_t = core_profiles_t
    self.core_profiles_t_plus_dt = core_profiles_t_plus_dt
    self.transport_model = transport_model
    self.explicit_source_profiles = explicit_source_profiles
    self.source_models = source_models
    self.evolving_names = evolving_names

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      x: tuple[fvm.CellVariable, ...],
      allow_pereverzev: bool = False,
      # Checks if reduced calc_coeffs for explicit terms when theta_imp=1
      # should be called
      explicit_call: bool = False,
  ):
    replace = {k: v for k, v in zip(self.evolving_names, x)}
    if explicit_call:
      core_profiles = config_args.recursive_replace(
          self.core_profiles_t, **replace
      )
    else:
      core_profiles = config_args.recursive_replace(
          self.core_profiles_t_plus_dt, **replace
      )
    # update ion density in core_profiles if ne is being evolved.
    # Necessary for consistency in iterative nonlinear solutions
    if 'ne' in self.evolving_names:
      ni = dataclasses.replace(
          core_profiles.ni,
          value=core_profiles.ne.value
          * physics.get_main_ion_dilution_factor(
              dynamic_runtime_params_slice.plasma_composition.Zimp,
              dynamic_runtime_params_slice.plasma_composition.Zeff,
          ),
      )
      core_profiles = dataclasses.replace(core_profiles, ni=ni)

    if allow_pereverzev:
      use_pereverzev = self.static_runtime_params_slice.stepper.use_pereverzev
    else:
      use_pereverzev = False

    return calc_coeffs.calc_coeffs(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=self.geo,
        core_profiles=core_profiles,
        transport_model=self.transport_model,
        explicit_source_profiles=self.explicit_source_profiles,
        source_models=self.source_models,
        evolving_names=self.evolving_names,
        use_pereverzev=use_pereverzev,
        explicit_call=explicit_call,
    )


class FrozenCoeffsCallback(CoeffsCallback):
  """CoeffsCallback that returns the same coefficients each time.

  NOTE: This class is mainly used for testing. It will ignore any time-dependent
  runtime configuration parameters, so it can yield incorrect results.
  """

  def __init__(self, *args, **kwargs):
    if 'dynamic_runtime_params_slice' not in kwargs:
      raise ValueError('dynamic_runtime_params_slice must be provided.')
    dynamic_runtime_params_slice = kwargs.pop('dynamic_runtime_params_slice')
    super().__init__(*args, **kwargs)
    x = tuple([self.core_profiles_t[name] for name in self.evolving_names])
    self.frozen_coeffs = super().__call__(
        dynamic_runtime_params_slice,
        x,
        allow_pereverzev=False,
        explicit_call=False,
    )

  def __call__(
      self,
      dynamic_runtime_params_slice,
      x,
      allow_pereverzev=False,
      explicit_call=False,
  ):

    return self.frozen_coeffs


class SimulationStepFn:
  """Advances the TORAX simulation one time step.

  Unlike the Stepper class, which updates certain parts of the state, a
  SimulationStepFn takes in the ToraxSimState and outputs the updated
  ToraxSimState, which contains not only the CoreProfiles but also extra
  simulation state useful for stepping as well as extra outputs useful for
  inspection inside the main run loop in `run_simulation()`. It wraps calls to
  Stepper with useful features to increase robustness for convergence, like
  dt-backtracking.
  """

  def __init__(
      self,
      stepper: stepper_lib.Stepper,
      time_step_calculator: ts.TimeStepCalculator,
      transport_model: transport_model_lib.TransportModel,
  ):
    """Initializes the SimulationStepFn.

    If you wish to run a simulation with new versions of any of these arguments
    (i.e. want to change to a new stepper), then you will need to build a new
    SimulationStepFn. These arguments are fixed for the lifetime
    of the SimulationStepFn and cannot change even with JAX recompiles.

    Args:
      stepper: Evolves the core profiles.
      time_step_calculator: Calculates the dt for each time step.
      transport_model: Calculates diffusion and convection coefficients.
    """
    self._stepper_fn = stepper
    self._time_step_calculator = time_step_calculator
    self._transport_model = transport_model
    self._jitted_transport_model = jax_utils.jit(
        transport_model.__call__,
    )

  @property
  def stepper(self) -> stepper_lib.Stepper:
    return self._stepper_fn

  @property
  def transport_model(self) -> transport_model_lib.TransportModel:
    return self._transport_model

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
      geo: geometry.Geometry,
      input_state: state.ToraxSimState,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
  ) -> state.ToraxSimState:
    """Advances the simulation state one time step.

    Args:
      static_runtime_params_slice: Static parameters that, if they change,
        should trigger a recompilation of the SimulationStepFn.
      dynamic_runtime_params_slice_provider: Object that returns a set of
        runtime parameters which may change from time step to time step or
        simulation run to run. If these runtime parameters change, it does NOT
        trigger a JAX recompilation.
      geo: The geometry of the torus during this time step of the simulation.
        While the geometry may change, any changes to the grid size can trigger
        recompilation of the stepper (if it is jitted) or an error (assuming it
        is JAX-compiled and lowered).
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      explicit_source_profiles: Explicit source profiles computed based on the
        core profiles at the start of the time step.

    Returns:
      ToraxSimState containing:
        - the core profiles at the end of the time step.
        - time and time step calculator state info.
        - core_sources and core_transport at the end of the time step.
        - stepper_error_state:
           0 if solver converged with fine tolerance for this step
           1 if solver did not converge for this step (was above coarse tol)
           2 if solver converged within coarse tolerance. Allowed to pass with a
             warning. Occasional error=2 has low impact on final sim state.
    """
    dynamic_runtime_params_slice_t = dynamic_runtime_params_slice_provider(
        input_state.t
    )
    # TODO(b/335598388): We call the transport model both here and in the the
    # Stepper / CoeffsCallback. This isn't a problem *so long as all of those
    # calls fall within the same jit scope* because can use
    # functools.lru_cache to avoid building duplicate expressions for the same
    # transport coeffs. We should still refactor the design to more explicitly
    # calculate transport coeffs at delta_t = 0 in only one place, so that we
    # have some flexibility in where to place the jit boundaries.
    transport_coeffs = self._jitted_transport_model(
        dynamic_runtime_params_slice_t, geo, input_state.core_profiles
    )

    # initialize new dt and reset stepper iterations.
    dt, time_step_calculator_state = self._time_step_calculator.next_dt(
        dynamic_runtime_params_slice_t,
        geo,
        input_state.core_profiles,
        input_state.time_step_calculator_state,
        transport_coeffs,
    )

    crosses_t_final = (
        input_state.t < dynamic_runtime_params_slice_t.numerics.t_final
    ) * (
        input_state.t + input_state.dt
        > dynamic_runtime_params_slice_t.numerics.t_final
    )
    dt = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice_t.numerics.exact_t_final,
            crosses_t_final,
        ),
        dynamic_runtime_params_slice_t.numerics.t_final - input_state.t,
        dt,
    )
    if jnp.any(jnp.isnan(dt)):
      raise ValueError('dt is NaN.')

    # The stepper needs the dynamic_runtime_params_slice at time t + dt for
    # implicit computations in the solver.
    dynamic_runtime_params_slice_t_plus_dt = (
        dynamic_runtime_params_slice_provider(
            input_state.t + dt,
        )
    )

    core_profiles_t = input_state.core_profiles

    # Construct the CoreProfiles object for time t+dt with evolving boundary
    # conditions and time-dependent prescribed profiles not directly solved by
    # PDE system.
    core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo=geo,
        core_profiles_t=core_profiles_t,
    )

    stepper_iterations = 0

    # Initial trial for stepper. If did not converge (can happen for nonlinear
    # step with large dt) we apply the adaptive time step routine if requested.
    core_profiles, core_sources, core_transport, stepper_error_state = (
        self._stepper_fn(
            dt=dt,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            geo=geo,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
        )
    )
    stepper_iterations += 1

    output_state = state.ToraxSimState(
        t=input_state.t + dt,
        dt=dt,
        core_profiles=core_profiles,
        core_transport=core_transport,
        core_sources=core_sources,
        stepper_iterations=stepper_iterations,
        time_step_calculator_state=time_step_calculator_state,
        stepper_error_state=stepper_error_state,
    )

    if static_runtime_params_slice.adaptive_dt:
      # Check if stepper converged. If not, proceed to body_fun
      def cond_fun(updated_output: state.ToraxSimState) -> bool:
        if updated_output.stepper_error_state == 1:
          do_dt_backtrack = True
        else:
          do_dt_backtrack = False
        return do_dt_backtrack

      # Make a new step with a smaller dt, starting with the original core
      # profiles.
      # Exit if dt < mindt
      def body_fun(
          updated_output: state.ToraxSimState,
      ) -> state.ToraxSimState:

        dt = (
            updated_output.dt
            / dynamic_runtime_params_slice_t.numerics.dt_reduction_factor
        )
        if jnp.any(jnp.isnan(dt)):
          raise ValueError('dt is NaN.')
        if dt < dynamic_runtime_params_slice_t.numerics.mindt:
          raise ValueError('dt below minimum timestep following adaptation')

        dynamic_runtime_params_slice_t_plus_dt = (
            dynamic_runtime_params_slice_provider(
                input_state.t + dt,
            )
        )
        core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
            core_profiles_t=core_profiles_t,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            static_runtime_params_slice=static_runtime_params_slice,
            geo=geo,
        )
        core_profiles, core_sources, core_transport, stepper_error_state = (
            self._stepper_fn(
                dt=dt,
                static_runtime_params_slice=static_runtime_params_slice,
                dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
                dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
                geo=geo,
                core_profiles_t=core_profiles_t,
                core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles,
            )
        )
        return dataclasses.replace(
            updated_output,
            t=input_state.t + dt,
            dt=dt,
            stepper_iterations=updated_output.stepper_iterations + 1,
            core_profiles=core_profiles,
            core_transport=core_transport,
            core_sources=core_sources,
            stepper_error_state=stepper_error_state,
        )

      output_state = jax_utils.py_while(cond_fun, body_fun, output_state)

    # Update total current, q, and s profiles based on new psi
    dynamic_runtime_params_slice_t_plus_dt = (
        dynamic_runtime_params_slice_provider(
            input_state.t + output_state.dt,
        )
    )
    q_corr = dynamic_runtime_params_slice_t_plus_dt.numerics.q_correction_factor
    output_state.core_profiles = physics.update_jtot_q_face_s_face(
        geo=geo,
        core_profiles=output_state.core_profiles,
        q_correction_factor=q_corr,
    )

    # Update ohmic and bootstrap current based on the new core profiles.
    output_state.core_profiles = update_current_distribution(
        source_models=self._stepper_fn.source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
        geo=geo,
        core_profiles=output_state.core_profiles,
    )

    # Update psidot based on the new core profiles
    output_state.core_profiles = update_psidot(
        source_models=self._stepper_fn.source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
        geo=geo,
        core_profiles=output_state.core_profiles,
    )

    return output_state


def get_initial_state(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    time_step_calculator: ts.TimeStepCalculator,
) -> state.ToraxSimState:
  """Returns the initial state to be used by run_simulation()."""
  initial_core_profiles = core_profile_setters.initial_core_profiles(
      dynamic_runtime_params_slice,
      geo,
      source_models,
  )
  return state.ToraxSimState(
      t=jnp.array(dynamic_runtime_params_slice.numerics.t_initial),
      dt=jnp.zeros(()),
      core_profiles=initial_core_profiles,
      # This will be overridden within run_simulation().
      core_sources=source_profiles_lib.SourceProfiles(
          profiles={},
          j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
              geo
          ),
          qei=source_profiles_lib.QeiInfo.zeros(geo),
      ),
      core_transport=state.CoreTransport.zeros(geo),
      time_step_calculator_state=time_step_calculator.initial_state(),
      stepper_error_state=0,
      stepper_iterations=0,
  )


class GeometryProvider(Protocol):
  """Returns the geometry to use during one time step of the simulation.

  A GeometryProvider is any callable (class or function) which takes the
  ToraxSimState at the start of a time step and returns the Geometry for that
  time step. See `run_simulation()` for how this callable is used.

  This class is a typing.Protocol, meaning it defines an interface, but any
  function asking for a GeometryProvider as an argument can accept any function
  or class that implements this API without specifically extending this class.

  For instance, the following is an equivalent implementation of the
  ConstantGeometryProvider without actually creating a class, and equally valid.

  .. code-block:: python

    geo = geometry.build_circular_geometry(...)
    constant_geo_provider = lamdba input_state: geo

    def func_expecting_geo_provider(gp: GeometryProvider):
      ... # do something with the provider.

    func_expecting_geo_provider(constant_geo_provider)  # this works.
  """

  def __call__(
      self,
      input_state: state.ToraxSimState,
  ) -> geometry.Geometry:
    """Returns the geometry to use during one time step of the simulation.

    The geometry may change from time step to time step, so the sim needs a
    callable to provide which geometry to use for a given time step (this is
    that callable). For most use cases, only the time will be relevant from the
    ToraxSimState (in order to support time-dependent geometries), but access to
    the full ToraxSimState is given to support any other use cases users may
    come up with.

    Args:
      input_state: Full simulation state at the start of the time step. This
        includes the time.

    Returns:
      Geometry of the torus to use for the time step.
    """


class ConstantGeometryProvider(GeometryProvider):
  """Returns the same Geometry for all calls."""

  def __init__(self, geo: geometry.Geometry):
    self._geo = geo

  def __call__(
      self,
      input_state: state.ToraxSimState,
  ) -> geometry.Geometry:
    # The API includes input_state as an arg even though it is unused in order
    # to match the API of a GeometryProvider.
    del input_state  # Ignored.
    return self._geo


# This class is read-only but not a frozen dataclass to allow us to set the
# SimulationStepFn attribute lazily when run() is called.
class Sim:
  """A lightweight object holding all components of a simulation.

  Use of this object is optional, it is also fine to hold these objects
  in local variables of a script and call `run_simulation` directly.

  The main purpose of the Sim object is to enable configuration via
  constructor arguments. Components are reused in subsequent simulation runs, so
  if a component is compiled, it will be reused for the next `Sim.run()` call
  and will not be recompiled unless a static argument or shape changes.
  """

  def __init__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
      geometry_provider: GeometryProvider,
      initial_state: state.ToraxSimState,
      time_step_calculator: ts.TimeStepCalculator,
      transport_model: transport_model_lib.TransportModel | None = None,
      stepper: stepper_lib.Stepper | None = None,
      step_fn: SimulationStepFn | None = None,
  ):
    self._static_runtime_params_slice = static_runtime_params_slice
    self._dynamic_runtime_params_slice_provider = (
        dynamic_runtime_params_slice_provider
    )
    self._geometry_provider = geometry_provider
    self._initial_state = initial_state
    self._time_step_calculator = time_step_calculator
    if step_fn is None:
      if stepper is None or transport_model is None:
        raise ValueError(
            'If step_fn is None, must provide both stepper and transport_model.'
        )
      self._stepper = stepper
      self._transport_model = transport_model
    else:
      ignored_params = [
          name
          for param, name in [
              (stepper, 'stepper'),
              (transport_model, 'transport_model'),
          ]
          if param is not None
      ]
      if ignored_params:
        logging.warning(
            'step_fn is not None, so the following parameters are ignored: %s',
            ignored_params,
        )
      self._stepper = step_fn.stepper
      self._transport_model = step_fn.transport_model
    self._step_fn = step_fn

  @property
  def time_step_calculator(self) -> ts.TimeStepCalculator:
    return self._time_step_calculator

  @property
  def initial_state(self) -> state.ToraxSimState:
    return self._initial_state

  @property
  def geometry_provider(self) -> GeometryProvider:
    return self._geometry_provider

  @property
  def dynamic_runtime_params_slice_provider(
      self,
  ) -> runtime_params_slice.DynamicRuntimeParamsSliceProvider:
    return self._dynamic_runtime_params_slice_provider

  @property
  def static_runtime_params_slice(
      self,
  ) -> runtime_params_slice.StaticRuntimeParamsSlice:
    return self._static_runtime_params_slice

  @property
  def step_fn(self) -> SimulationStepFn | None:
    return self._step_fn

  @property
  def stepper(self) -> stepper_lib.Stepper:
    return self._stepper

  @property
  def transport_model(self) -> transport_model_lib.TransportModel:
    return self._transport_model

  @property
  def source_models(self) -> source_models_lib.SourceModels:
    if self._step_fn is None:
      assert self._stepper is not None
      return self._stepper.source_models
    return self._step_fn.stepper.source_models

  def run(
      self,
      log_timestep_info: bool = False,
      spectator: spectator_lib.Spectator | None = None,
  ) -> tuple[state.ToraxSimState, ...]:
    """Runs the transport simulation over a prescribed time interval.

    See `run_simulation` for details.

    Args:
      log_timestep_info: See `run_simulation()`.
      spectator: If a SimulationStepFn has not yet been built for this Sim
        object (if it was not passed in __init__ or this object has never been
        run), then it will be built in this call, and this spectator will be
        built into it. If the SimulationStepFn has already been built, then this
        argument is ignored and the spectator built into the SimulationStepFn
        cannot change. In these cases where you want to use a new spectator, you
        must build a new Sim object.

    Returns:
      Tuple of all ToraxSimStates, one per time step and an additional one at
      the beginning for the starting state.
    """
    if self._step_fn is None:
      self._step_fn = SimulationStepFn(
          stepper=self._stepper,
          time_step_calculator=self.time_step_calculator,
          transport_model=self._transport_model,
      )
    assert self.step_fn
    if spectator is not None:
      spectator.reset()
    return run_simulation(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
        geometry_provider=self.geometry_provider,
        initial_state=self.initial_state,
        time_step_calculator=self.time_step_calculator,
        step_fn=self.step_fn,
        log_timestep_info=log_timestep_info,
        spectator=spectator,
    )


def build_sim_object(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geo: geometry.Geometry,
    stepper_builder: stepper_lib.StepperBuilder,
    transport_model_builder: transport_model_lib.TransportModelBuilder,
    source_models: source_models_lib.SourceModels,
    time_step_calculator: Optional[ts.TimeStepCalculator] = None,
) -> Sim:
  """Builds a Sim object from the input runtime params and sim components.

  The Sim object provides a container for all the components that go into a
  single TORAX simulation run. It gives a way to reuse components without having
  to rebuild or recompile them if JAX shapes or static arguments do not change.

  Read more about the Sim object in its class docstring. The use of it is
  optional, and users may call `sim.run_simulation()` directly as well.

  Args:
    runtime_params: The input runtime params used throughout the simulation run.
    geo: Describes the magnetic geometry.
    stepper_builder: A callable to build the stepper. The stepper has already
      been factored out of the config.
    transport_model_builder: A callable to build the transport model.
    source_models: All TORAX sources/sink functions which provide profiles used
      as terms in the equations that evolve the core profiless.
    time_step_calculator: The time_step_calculator, if built, otherwise a
      ChiTimeStepCalculator will be built by default.

  Returns:
    sim: The built Sim instance.
  """

  transport_model = transport_model_builder()

  static_runtime_params_slice = (
      runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          stepper=stepper_builder.runtime_params,
      )
  )
  dynamic_runtime_params_slice_provider = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport_getter=lambda: transport_model_builder.runtime_params,
          sources_getter=lambda: source_models.runtime_params,
          stepper_getter=lambda: stepper_builder.runtime_params,
      )
  )
  stepper = stepper_builder(transport_model, source_models)

  if time_step_calculator is None:
    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()

  # build dynamic_runtime_params_slice at t_initial for initial conditions
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      runtime_params.numerics.t_initial
  )
  initial_state = get_initial_state(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      source_models=stepper.source_models,
      time_step_calculator=time_step_calculator,
  )

  return Sim(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
      geometry_provider=ConstantGeometryProvider(geo),
      initial_state=initial_state,
      time_step_calculator=time_step_calculator,
      transport_model=transport_model,
      stepper=stepper,
  )


def run_simulation(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
    geometry_provider: GeometryProvider,
    initial_state: state.ToraxSimState,
    time_step_calculator: ts.TimeStepCalculator,
    step_fn: SimulationStepFn,
    log_timestep_info: bool = False,
    spectator: spectator_lib.Spectator | None = None,
) -> tuple[state.ToraxSimState, ...]:
  """Runs the transport simulation over a prescribed time interval.

  This is the main entrypoint for running a TORAX simulation.

  This function runs a variable number of time steps until the
  time_step_calculator determines the sim is done, using a Python while loop.

  Args:
    static_runtime_params_slice: A static set of arguments to provide to the
      step_fn. If internal functions in step_fn are JAX-compiled, then these
      params are "compile-time constant" meaning that they are considered static
      to the compiled functions. If they change (i.e. the same step_fn is called
      again with a different static_runtime_params_slice), then internal
      functions will be recompiled. JAX determines if recompilation is necessary
      via the hash of static_runtime_params_slice.
    dynamic_runtime_params_slice_provider: Provides a DynamicRuntimeParamsSlice
      to use as input for each time step. See static_runtime_params_slice and
      the runtime_params_slice module docstring for runtime_params_slice to
      understand why we need the dynamic and static config slices and what they
      control.
    geometry_provider: Provides the magnetic geometry for each time step
      based on the ToraxSimState at the start of the time step. The geometry may
      change from time step to time step, so the sim needs a function to provide
      which geometry to use for a given time step. A GeometryProvider is any
      callable (class or function) which takes the ToraxSimState at the start of
      a time step and returns the Geometry for that time step. For most use
      cases, only the time will be relevant from the ToraxSimState (in order to
      support time-dependent geometries).
    initial_state: The starting state of the simulation. This includes both the
      state variables which the stepper.Stepper will evolve (like ion temp, psi,
      etc.) as well as other states that need to be be tracked, like time.
    time_step_calculator: TimeStepCalculator determining policy for stepping
      through time.
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    spectator: Object which can "spectate" values as the simulation runs. See
      the Spectator class docstring for more details.

  Returns:
    tuple of ToraxSimState objects, one for each time step. There are N+1
    objects returned, where N is the number of simulation steps taken. The first
    object in the tuple is for the initial state.
  """

  # Provide logging information on precision setting
  if jax.config.read('jax_enable_x64'):
    logging.info('Precision is set at float64')
  else:
    logging.info('Precision is set at float32')

  logging.info('Starting simulation.')
  # Python while loop implementation.
  # Not efficient for grad, jit of grad.
  # Uses time_step_calculator.not_done to decide when to stop.
  # Note: can't use a jax while loop due to appending to history.

  running_main_loop_start_time = time.time()
  wall_clock_step_times = []
  torax_outputs = [
      initial_state,
  ]
  stepper_error_state = 0
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      initial_state.t
  )
  geo = geometry_provider(initial_state)

  # Populate the starting state with source profiles from the implicit sources
  # before starting the run-loop. The explicit source profiles will be computed
  # inside the loop and will be merged with these implicit source profiles.
  initial_state.core_sources = _get_initial_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=initial_state.core_profiles,
      source_models=step_fn.stepper.source_models,
  )
  if spectator is not None:
    # Because of the updates we apply to the core sources during the next
    # iteration, we need to start the spectator before step here.
    spectator.before_step()

  sim_state = initial_state
  # Keep advancing the simulation until the time_step_calculator tells us we are
  # done.
  while time_step_calculator.not_done(
      sim_state.t,
      dynamic_runtime_params_slice,
      sim_state.time_step_calculator_state,
  ):
    # Measure how long in wall clock time each simulation step takes.
    step_start_time = time.time()
    if log_timestep_info:
      _log_timestep(sim_state.t, sim_state.dt, sim_state.stepper_iterations)
      # TODO(b/330172917): once tol and coarse_tol are configurable in the
      # runtime_params, also log the value of tol and coarse_tol below
      match stepper_error_state:
        case 0:
          pass
        case 1:
          logging.info('Solver did not converge in previous step.')
        case 2:
          logging.info(
              'Solver converged only within coarse tolerance in previous step.'
          )

    # This only computes sources set to explicit in the
    # DynamicSourceConfigSlice. All implicit sources will have their profiles
    # set to 0.
    explicit_source_profiles = source_models_lib.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=sim_state.core_profiles,
        source_models=step_fn.stepper.source_models,
        explicit=True,
    )

    # The previous time step's state has an incomplete set of source profiles
    # which was computed based on the previous time step's "guess" of the core
    # profiles at this time step's t. We can merge those "implicit" source
    # profiles with the explicit ones computed here.
    sim_state.core_sources = _merge_source_profiles(
        explicit_source_profiles=explicit_source_profiles,
        implicit_source_profiles=sim_state.core_sources,
        source_models=step_fn.stepper.source_models,
        qei_core_profiles=sim_state.core_profiles,
    )
    # Make sure to "spectate" the state after the source profiles  have been
    # merged and updated in the output sim_state.
    if spectator is not None:
      _update_spectator(spectator, sim_state)
      # This is after the previous time step's step_fn() call.
      spectator.after_step()
      # Now prep the spectator for the following time step.
      spectator.before_step()
    sim_state = step_fn(
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geo,
        sim_state,
        explicit_source_profiles,
    )
    stepper_error_state = sim_state.stepper_error_state
    # Update the runtime config for the next iteration.
    dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
        sim_state.t
    )
    torax_outputs.append(sim_state)
    geo = geometry_provider(sim_state)
    wall_clock_step_times.append(time.time() - step_start_time)
  # Log final timestep
  if log_timestep_info:
    # The "sim_state" here has been updated by the loop above.
    _log_timestep(sim_state.t, sim_state.dt, sim_state.stepper_iterations)

  # Update the final time step's source profiles based on the explicit source
  # profiles computed based on the final state.
  logging.info("Updating last step's source profiles.")
  explicit_source_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=sim_state.core_profiles,
      source_models=step_fn.stepper.source_models,
      explicit=True,
  )
  sim_state.core_sources = _merge_source_profiles(
      explicit_source_profiles=explicit_source_profiles,
      implicit_source_profiles=sim_state.core_sources,
      source_models=step_fn.stepper.source_models,
      qei_core_profiles=sim_state.core_profiles,
  )
  if spectator is not None:
    # Complete the last time step.
    _update_spectator(spectator, sim_state)
    spectator.after_step()

  # If the first step of the simulation was very long, call it out. It might
  # have to do with tracing the jitted step_fn.
  std_devs = 2  # Check if the first step is more than 2 std devs longer.
  if wall_clock_step_times and wall_clock_step_times[0] > (
      jnp.mean(jnp.array(wall_clock_step_times))
      + std_devs * jnp.std(jnp.array(wall_clock_step_times))
  ):
    long_first_step = True
    logging.info(
        'The first step took more than %.1f std devs longer than other steps.'
        ' It likely was tracing and compiling the step_fn. It took %.2f '
        'seconds of wall clock time.',
        std_devs,
        wall_clock_step_times[0],
    )
  else:
    long_first_step = False

  wall_clock_time_elapsed = time.time() - running_main_loop_start_time
  simulation_time = torax_outputs[-1].t - torax_outputs[0].t
  if long_first_step:
    # Don't include the long first step in the total time logged.
    wall_clock_time_elapsed -= wall_clock_step_times[0]
  logging.info(
      'Simulated %.2f seconds of physics in %.2f seconds of wall clock time.',
      simulation_time,
      wall_clock_time_elapsed,
  )
  return tuple(torax_outputs)


def _update_spectator(
    spectator: spectator_lib.Spectator,
    output_state: state.ToraxSimState,
) -> None:
  """Updates the spectator with values from the output state."""
  spectator.observe(key='q_face', data=output_state.core_profiles.q_face)
  spectator.observe(key='s_face', data=output_state.core_profiles.s_face)
  spectator.observe(key='ne', data=output_state.core_profiles.ne.value)
  spectator.observe(
      key='temp_ion',
      data=output_state.core_profiles.temp_ion.value,
  )
  spectator.observe(
      key='temp_el',
      data=output_state.core_profiles.temp_el.value,
  )
  spectator.observe(
      key='j_bootstrap_face',
      data=output_state.core_profiles.currents.j_bootstrap_face,
  )
  spectator.observe(
      key='johm_face',
      data=output_state.core_profiles.currents.johm_face,
  )
  spectator.observe(
      key='jext_face',
      data=output_state.core_profiles.currents.jext_face,
  )
  spectator.observe(
      key='jtot_face',
      data=output_state.core_profiles.currents.jtot_face,
  )
  spectator.observe(
      key='chi_face_ion', data=output_state.core_transport.chi_face_ion
  )
  spectator.observe(
      key='chi_face_el', data=output_state.core_transport.chi_face_el
  )
  spectator.observe(
      key='Qext_i',
      data=output_state.core_sources.get_profile(
          'generic_ion_el_heat_source_ion'
      ),
  )
  spectator.observe(
      key='Qext_e',
      data=output_state.core_sources.get_profile(
          'generic_ion_el_heat_source_el'
      ),
  )
  spectator.observe(
      key='Qfus_i',
      data=output_state.core_sources.get_profile('fusion_heat_source_ion'),
  )
  spectator.observe(
      key='Qfus_e',
      data=output_state.core_sources.get_profile('fusion_heat_source_el'),
  )
  spectator.observe(
      key='Qohm',
      data=output_state.core_sources.get_profile('ohmic_heat_source'),
  )
  spectator.observe(
      key='Qei', data=output_state.core_sources.get_profile('qei_source')
  )


def update_current_distribution(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Update bootstrap current based on the new core_profiles."""

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ],
      geo=geo,
      core_profiles=core_profiles,
  )

  johm = (
      core_profiles.currents.jtot
      - bootstrap_profile.j_bootstrap
      - core_profiles.currents.jext
  )
  johm_face = (
      core_profiles.currents.jtot_face
      - bootstrap_profile.j_bootstrap_face
      - core_profiles.currents.jext_face
  )

  currents = dataclasses.replace(
      core_profiles.currents,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      johm=johm,
      johm_face=johm_face,
  )
  new_core_profiles = dataclasses.replace(
      core_profiles,
      currents=currents,
  )
  return new_core_profiles


def update_psidot(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Update psidot based on new core_profiles."""

  psidot = dataclasses.replace(
      core_profiles.psidot,
      value=source_models_lib.calc_psidot(
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
      ),
  )

  new_core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
  )
  return new_core_profiles


def provide_core_profiles_t_plus_dt(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
  """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
  updated_boundary_conditions = (
      core_profile_setters.compute_boundary_conditions(
          dynamic_runtime_params_slice_t_plus_dt,
          geo,
      )
  )
  updated_values = core_profile_setters.updated_prescribed_core_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
      geo=geo,
      core_profiles=core_profiles_t,
  )
  temp_ion = dataclasses.replace(
      core_profiles_t.temp_ion,
      value=updated_values['temp_ion'],
      **updated_boundary_conditions['temp_ion'],
  )
  temp_el = dataclasses.replace(
      core_profiles_t.temp_el,
      value=updated_values['temp_el'],
      **updated_boundary_conditions['temp_el'],
  )
  psi = dataclasses.replace(
      core_profiles_t.psi, **updated_boundary_conditions['psi']
  )
  ne = dataclasses.replace(
      core_profiles_t.ne,
      value=updated_values['ne'],
      **updated_boundary_conditions['ne'],
  )
  ni = dataclasses.replace(
      core_profiles_t.ni,
      value=updated_values['ni'],
      **updated_boundary_conditions['ni'],
  )
  core_profiles_t_plus_dt = dataclasses.replace(
      core_profiles_t, temp_ion=temp_ion, temp_el=temp_el, psi=psi, ne=ne, ni=ni
  )
  return core_profiles_t_plus_dt


def _get_initial_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> source_profiles_lib.SourceProfiles:
  """Returns the "implicit" profiles for the initial state in run_simulation().

  The source profiles returned as part of each time step's state in
  run_simulation() is computed based on the core profiles at that time step.
  However, for the first time step, only the explicit sources are computed.
  The
  implicit profiles are computed based on the "live" state that is evolving,
  "which depending on the stepper used is either consistent (within tolerance)
  with the state at the next timestep (nonlinear solver), or an approximation
  thereof (linear stepper or predictor-corrector)." So for the first time step,
  we need to prepopulate the state with the implicit profiles for the starting
  core profiles.

  Args:
    static_runtime_params_slice: Runtime parameters which, when they change,
      trigger recompilations. They should not change within a single run of the
      sim.
    dynamic_runtime_params_slice: Runtime parameters which may change from time
      step to time step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.

  Returns:
    SourceProfiles from implicit source models based on the core profiles from
    the starting state.
  """
  implicit_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=False,
  )
  qei = source_models.qei_source.get_qei(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.qei_source_name
      ],
      geo=geo,
      core_profiles=core_profiles,
  )
  implicit_profiles = dataclasses.replace(implicit_profiles, qei=qei)
  return implicit_profiles


# This function can be jitted if source_models is a static argument. However,
# in our tests, jitting this function actually slightly slows down runs, so this
# is left as pure python.
def _merge_source_profiles(
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    implicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    qei_core_profiles: state.CoreProfiles,
) -> source_profiles_lib.SourceProfiles:
  """Returns a SourceProfiles that merges the input profiles.

  Sources can either be explicit or implicit. The explicit_source_profiles
  contain the profiles for all source models that are set to explicit, and it
  contains profiles with all zeros for any implicit source. The opposite holds
  for the implicit_source_profiles.

  This function adds the two dictionaries of profiles and returns a single
  SourceProfiles that includes both.

  Args:
    explicit_source_profiles: Profiles from explicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the explicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.
    implicit_source_profiles: Profiles from implicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the implicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.
    source_models: Source models used to compute the profiles given.
    qei_core_profiles: The core profiles used to compute the Qei source.

  Returns:
    A SourceProfiles with non-zero profiles for all sources, both explicit and
    implicit (assuming the source model outputted a non-zero profile).
  """
  sum_profiles = lambda a, b: a + b
  summed_bootstrap_profile = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.j_bootstrap,
      implicit_source_profiles.j_bootstrap,
  )
  summed_qei_info = jax.tree_util.tree_map(
      sum_profiles, explicit_source_profiles.qei, implicit_source_profiles.qei
  )
  summed_other_profiles = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.profiles,
      implicit_source_profiles.profiles,
  )
  # For ease of comprehension, we convert the units of the Qei source and add it
  # to the list of other profiles before returning it.
  summed_other_profiles[source_models.qei_source_name] = (
      summed_qei_info.qei_coef
      * (qei_core_profiles.temp_el.value - qei_core_profiles.temp_ion.value)
  )
  # Also for better comprehension of the outputs, any known source model output
  # profiles that contain both the ion and electron heat profiles (i.e. they are
  # 2D with the separate profiles stacked) are unstacked here.
  for source_name in source_models.ion_el_sources:
    if source_name in summed_other_profiles:
      # The profile in summed_other_profiles is 2D. We want to split it up.
      if (
          f'{source_name}_ion' in summed_other_profiles
          or f'{source_name}_el' in summed_other_profiles
      ):
        raise ValueError(
            'Source model names are too close. Trying to save the output from '
            f'the source {source_name} in 2 components: {source_name}_ion and '
            f'{source_name}_el, but there is already a source output with that '
            'name. Rename your sources to avoid this collision.'
        )
      summed_other_profiles[f'{source_name}_ion'] = summed_other_profiles[
          source_name
      ][0, ...]
      summed_other_profiles[f'{source_name}_el'] = summed_other_profiles[
          source_name
      ][1, ...]
      del summed_other_profiles[source_name]
  return source_profiles_lib.SourceProfiles(
      profiles=summed_other_profiles,
      j_bootstrap=summed_bootstrap_profile,
      qei=summed_qei_info,
  )
