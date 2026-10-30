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

"""Pydantic config for time step calculators."""

import abc
import math
from typing import Annotated, Final, Literal, Self

import chex
import numpy as np
import pydantic
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import from_previous_time_step_calculator
from torax._src.time_step_calculator import runtime_params
from torax._src.time_step_calculator import time_step_calculator
from torax._src.torax_pydantic import torax_pydantic

_BACKTRACKING_MAX_STEPS_MULTIPLIER: Final[int] = 10


class TimeStepCalculatorConfigBase(torax_pydantic.ConfigBase, abc.ABC):
  """Base config for a time step calculator.

  Attributes:
    t_initial: Simulation start time [s].
    t_final: Simulation end time [s].
    tolerance: The tolerance within the final time for which the simulation will
      be considered done.
    exact_t_final: If True, ensures that the simulation end time is exactly
      `t_final`, by adapting the final `dt` to match.
    min_dt: Minimum timestep allowed in simulation [s].
    dt_backtracking_factor: Factor by which to reduce dt if
      `allow_backtracking` is True or for `from_previous_dt` time step
      calculation.
    allow_backtracking: Iterative reduction of dt if solver fails to converge.
      If a time step does not converge, then the step is redone
      iteratively at successively lower dt until convergence is reached.
    max_steps: Optional maximum number of simulation steps. If None, falls back
      to `default_max_steps` if supported by the time step calculator.
  """

  t_initial: torax_pydantic.Second = 0.0
  t_final: torax_pydantic.Second = 5.0
  tolerance: float = 1e-7
  exact_t_final: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  min_dt: torax_pydantic.Second = 1e-8
  dt_backtracking_factor: pydantic.PositiveFloat = 3.0
  allow_backtracking: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  max_steps: Annotated[
      pydantic.PositiveInt | None, torax_pydantic.JAX_STATIC
  ] = None

  @pydantic.model_validator(mode='after')
  def _validate_base(self) -> Self:
    if self.t_initial > self.t_final:
      raise ValueError(
          't_initial must be less than or equal to t_final. '
          f't_initial: {self.t_initial}, t_final: {self.t_final}'
      )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    del t
    return runtime_params.RuntimeParams(
        t_initial=self.t_initial,
        t_final=self.t_final,
        tolerance=self.tolerance,
        min_dt=self.min_dt,
        dt_backtracking_factor=self.dt_backtracking_factor,
        exact_t_final=self.exact_t_final,
        allow_backtracking=self.allow_backtracking,
    )

  @property
  def default_max_steps(self) -> int:
    """Returns the default maximum number of simulation steps.

    Raises:
      NotImplementedError: If a default `max_steps` cannot be determined for
        this time step calculator.
    """
    # TODO(b/564475000): Support max_steps heuristics for other time step
    # calculators.
    raise NotImplementedError(
        f'max_steps must be specified when using {type(self).__name__}.'
    )

  @abc.abstractmethod
  def build_time_step_calculator(
      self,
  ) -> time_step_calculator.TimeStepCalculator:
    """Builds a time step calculator from the config."""


class ChiTimeStepCalculatorConfig(TimeStepCalculatorConfigBase):
  """Config for the chi time step calculator.

  Computes the candidate timestep `dt` at each step by scaling a base explicit
  parabolic stability limit derived from the maximum total heat diffusivity
  `chi_max = max(chi_i, chi_e)` across the radial grid:

    `basic_dt = (3 / 4) * min(drho_norm)**2 / chi_max`
    `dt = min(chi_timestep_prefactor * basic_dt, max_dt)`

  When `exact_t_final=True`, the final step is clipped to `t_final - t`.
  If `allow_backtracking=True` and the solver fails to converge, the
  current step is retried at `dt / dt_backtracking_factor**i` until either
  convergence is reached or `dt <= min_dt`. The next step will then return to
  the `dt` given by the above equation.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'chi'.
    max_dt: Maximum timestep allowed in the simulation.
    chi_timestep_prefactor: Prefactor multiplying the base explicit stability
      timestep `basic_dt`. For implicit solvers, this can typically be set to a
      large value to allow a larger timestep.
  """

  calculator_type: Annotated[Literal['chi'], torax_pydantic.JAX_STATIC] = 'chi'
  max_dt: torax_pydantic.Second = 2.0
  chi_timestep_prefactor: pydantic.PositiveFloat = 50.0

  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if not self.max_dt >= self.min_dt:
      raise ValueError(
          'max_dt must be greater than or equal to min_dt. '
          f'max_dt: {self.max_dt}, min_dt: {self.min_dt}'
      )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.ChiRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.ChiRuntimeParams(
        **vars(base_params),
        max_dt=self.max_dt,
        chi_timestep_prefactor=self.chi_timestep_prefactor,
    )

  def build_time_step_calculator(
      self,
  ) -> chi_time_step_calculator.ChiTimeStepCalculator:
    return chi_time_step_calculator.ChiTimeStepCalculator()


class FixedTimeStepCalculatorConfig(TimeStepCalculatorConfigBase):
  """Config for the fixed time step calculator.

  Steps through time using the prescribed `dt` evaluated at the current
  simulation time `t` (either a constant scalar or a step-interpolated
  time-varying scalar).

  When `exact_t_final=True`, the final step is clipped to `t_final - t`.
  If `allow_backtracking=True` and the solver fails to converge, the
  current step is retried at `dt / dt_backtracking_factor**i` until either
  convergence is reached or `dt <= min_dt`. The next step will then return to
  the prescribed `dt`.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'fixed'.
    dt: Prescribed timestep in seconds. May be a constant scalar or a
      step-interpolated time-varying scalar evaluated at the current simulation
      time `t`; actual steps may be smaller when `allow_backtracking=True` or
      on the final step when `exact_t_final=True`.
  """

  calculator_type: Annotated[Literal['fixed'], torax_pydantic.JAX_STATIC] = (
      'fixed'
  )
  dt: torax_pydantic.NonNegativeTimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(1e-1)
  )

  @property
  def default_max_steps(self) -> int:
    """Returns the default maximum number of simulation steps."""
    min_dt_val = float(np.min(self.dt.value))
    max_steps = math.ceil((self.t_final - self.t_initial) / min_dt_val)
    if self.allow_backtracking:
      max_steps *= _BACKTRACKING_MAX_STEPS_MULTIPLIER
    return max_steps

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.FixedRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.FixedRuntimeParams(
        **vars(base_params),
        dt=self.dt.get_value(t),
    )

  def build_time_step_calculator(
      self,
  ) -> fixed_time_step_calculator.FixedTimeStepCalculator:
    return fixed_time_step_calculator.FixedTimeStepCalculator()


class FromPreviousTimeStepCalculatorConfig(TimeStepCalculatorConfigBase):
  """Config for the from_previous_dt time step calculator.

  Stateful time step calculator that attempts to increase the timestep `dt` at
  each step, based on the most recent non-sawtooth timestep
  `previous_non_sawtooth_dt`:

    `dt = min(previous_non_sawtooth_dt * dt_backtracking_factor, max_dt)`

  On the first (non-sawtooth) step, it uses `initial_dt` (clipped to `max_dt`).
  When `exact_t_final=True`, the final step is clipped to `t_final - t`.
  Requires `allow_backtracking=True`.
  If the solver fails to converge, the current step is retried at
  `dt / dt_backtracking_factor**i` until either convergence is reached or
  `dt <= min_dt`. The next step will then calculate `dt` from the timestep that
  was accepted for this step.

  Attributes:
    calculator_type: The type of time step calculator to use, hardcoded to
      'from_previous_dt'.
    initial_dt: Initial timestep duration used for the first step.
    max_dt: Maximum timestep allowed in the simulation.
  """

  calculator_type: Annotated[
      Literal['from_previous_dt'],
      torax_pydantic.JAX_STATIC,
  ] = 'from_previous_dt'
  initial_dt: pydantic.PositiveFloat = 1e-1
  max_dt: torax_pydantic.Second = 2.0

  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if not self.allow_backtracking:
      raise ValueError(
          'allow_backtracking must be True when using from_previous_dt.'
      )
    if not self.max_dt >= self.min_dt:
      raise ValueError(
          'max_dt must be greater than or equal to min_dt. '
          f'max_dt: {self.max_dt}, min_dt: {self.min_dt}'
      )
    if not self.max_dt >= self.initial_dt:
      raise ValueError(
          'max_dt must be greater than or equal to initial_dt. '
          f'max_dt: {self.max_dt}, initial_dt: {self.initial_dt}'
      )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.FromPreviousRuntimeParams:
    base_params = super().build_runtime_params(t)
    return runtime_params.FromPreviousRuntimeParams(
        **vars(base_params),
        initial_dt=self.initial_dt,
        max_dt=self.max_dt,
    )

  def build_time_step_calculator(
      self,
  ) -> from_previous_time_step_calculator.FromPreviousTimeStepCalculator:
    return from_previous_time_step_calculator.FromPreviousTimeStepCalculator()


TimeStepCalculatorConfig = Annotated[
    ChiTimeStepCalculatorConfig
    | FixedTimeStepCalculatorConfig
    | FromPreviousTimeStepCalculatorConfig,
    pydantic.Field(discriminator='calculator_type'),
]
