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

"""PID feedback particle source for the electron density (n_e) equation.

This module implements a particle source (`generic_particle` with
`model_name='feedback'`) for electron density that combines a feedforward
particle injection term with a PID feedback control loop. The feedback loop
adjusts the total particle fueling rate to track a prescribed line-averaged or
volume-averaged electron density target, and deposits particles radially using a
configurable profile shape (`'exponential'`, `'gaussian'`, or `'prescribed'`).
"""

import dataclasses
import enum
from typing import Annotated, Literal, Self, TypeAlias

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import solver_profiles as solver_profiles_lib
from torax._src.geometry import geometry
from torax._src.physics import formulas as physics_formulas
from torax._src.sources import base
from torax._src.sources import core_sources
from torax._src.sources import formulas
from torax._src.sources import generic_particle_source
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


class AverageType(enum.StrEnum):
  """Method used for spatial averaging of electron density."""

  LINE = 'line'
  VOLUME = 'volume'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExponentialProfileRuntimeParams:
  """Runtime parameters for an exponential radial particle profile."""

  width: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GaussianProfileRuntimeParams:
  """Runtime parameters for a Gaussian radial particle profile."""

  center: array_typing.FloatScalar
  width: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrescribedProfileRuntimeParams:
  """Runtime parameters for a prescribed radial particle profile shape."""

  values: array_typing.FloatVectorCell


ProfileRuntimeParams: TypeAlias = (
    ExponentialProfileRuntimeParams
    | GaussianProfileRuntimeParams
    | PrescribedProfileRuntimeParams
)


class ExponentialProfileConfig(torax_pydantic.ConfigBase):
  """Exponential radial profile shape for `ParticleFeedbackSourceConfig`.

  Parameterized as `C * exp(-(1.0 - rho_norm) / width)`, where `C` is chosen so
  the volume integral equals `S_total`.

  Attributes:
    profile_type: Discriminator literal `'exponential'`.
    width: Exponential decay length from the edge inwards [$rho_norm$].
  """

  profile_type: Annotated[Literal['exponential'], torax_pydantic.JAX_STATIC] = (
      'exponential'
  )
  width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> ExponentialProfileRuntimeParams:
    """Builds runtime parameters evaluated at time `t`."""
    return ExponentialProfileRuntimeParams(
        width=self.width.get_value(t),
    )


class GaussianProfileConfig(torax_pydantic.ConfigBase):
  """Gaussian radial profile shape for `ParticleFeedbackSourceConfig`.

  Parameterized as `C * exp(-(rho_norm - center)^2 / (2 * width^2))`, where `C`
  is chosen so the volume integral equals `S_total`.

  Attributes:
    profile_type: Discriminator literal `'gaussian'`.
    center: Gaussian center in $[0, 1]$ [$rho_norm$].
    width: Gaussian standard deviation [$rho_norm$].
  """

  profile_type: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
  center: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.25)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> GaussianProfileRuntimeParams:
    """Builds runtime parameters evaluated at time `t`."""
    return GaussianProfileRuntimeParams(
        center=self.center.get_value(t),
        width=self.width.get_value(t),
    )


class PrescribedProfileConfig(torax_pydantic.ConfigBase):
  """Prescribed radial profile shape for `ParticleFeedbackSourceConfig`.

  Rescales a non-negative radial shape array on the cell grid so that its volume
  integral equals `S_total`. If the shape integrates to zero at time `t`, the
  deposited particle profile is identically zero everywhere.

  Attributes:
    profile_type: Discriminator literal `'prescribed'`.
    values: Non-negative unnormalized radial deposition shape interpolated onto
      the cell grid [arbitrary non-negative units].
  """

  profile_type: Annotated[Literal['prescribed'], torax_pydantic.JAX_STATIC] = (
      'prescribed'
  )
  values: torax_pydantic.NonNegativeTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 1.0, 1.0: 1.0}})
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> PrescribedProfileRuntimeParams:
    """Builds runtime parameters evaluated at time `t`."""
    return PrescribedProfileRuntimeParams(
        values=self.values.get_value(t),
    )


ProfileConfig: TypeAlias = Annotated[
    ExponentialProfileConfig | GaussianProfileConfig | PrescribedProfileConfig,
    pydantic.Discriminator('profile_type'),
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  """Runtime parameters for `ParticleFeedbackSource`.

  Attributes:
    profile: Evaluated radial deposition profile shape parameters.
    S_feedforward: Baseline feedforward particle injection rate [particles/s].
    target_average_n_e: Target average electron density [m^-3].
    average_type: Spatial averaging method (`LINE` or `VOLUME`).
    proportional_gain: Proportional feedback gain [m^3 / s].
    integral_gain: Integral feedback gain [m^3 / s^2].
    derivative_gain: Derivative feedback gain [m^3].
    derivative_smoothing_time_scale: Exponential smoothing time scale [s] for
      the error time derivative.
    min_integral: Lower clamping bound [particles/s] on the integral term.
    max_integral: Upper clamping bound [particles/s] on the integral term.
  """

  profile: ProfileRuntimeParams
  S_feedforward: array_typing.FloatScalar
  target_average_n_e: array_typing.FloatScalar
  average_type: AverageType = jax.tree.static()
  proportional_gain: array_typing.FloatScalar
  integral_gain: array_typing.FloatScalar
  derivative_gain: array_typing.FloatScalar
  derivative_smoothing_time_scale: array_typing.FloatScalar
  min_integral: array_typing.FloatScalar
  max_integral: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ParticleFeedbackState(source_state_lib.SourceState):
  """Cross-timestep PID controller state for ParticleFeedbackSource.

  Attributes:
    integral_term: Accumulated integral fueling contribution [particles/s],
      clamped to `[min_integral, max_integral]`.
    previous_error: Density tracking error [m^-3] at time `t`.
    smoothed_derivative: Exponentially smoothed time derivative of density error
      `de/dt` [m^-3 / s] at time `t`.
    t: Simulation time [s] at which this state was evaluated.
    initialized: Whether `previous_error` has been populated from an initial
      evaluation.
  """

  integral_term: array_typing.FloatScalar
  previous_error: array_typing.FloatScalar
  smoothed_derivative: array_typing.FloatScalar
  t: array_typing.FloatScalar
  initialized: array_typing.BoolScalar


def _compute_radial_profile(
    profile_params: ProfileRuntimeParams,
    S_total: array_typing.FloatScalar,
    geo: geometry.Geometry,
) -> array_typing.FloatVectorCell:
  """Computes the cell-grid particle source profile with integral `S_total`."""
  if isinstance(profile_params, ExponentialProfileRuntimeParams):
    return formulas.exponential_profile(
        geo=geo,
        decay_start=1.0,
        width=profile_params.width,
        total=S_total,
    )
  if isinstance(profile_params, GaussianProfileRuntimeParams):
    return formulas.gaussian_profile(
        geo=geo,
        center=profile_params.center,
        width=profile_params.width,
        total=S_total,
    )
  if isinstance(profile_params, PrescribedProfileRuntimeParams):
    shape = profile_params.values
    integrated = math_utils.volume_integration(shape, geo)
    # Guard denominator inside jnp.where to avoid NaN reverse-mode gradients
    # when the prescribed shape is identically zero.
    safe_integrated = jnp.where(integrated > 0.0, integrated, 1.0)
    return jnp.where(
        integrated > 0.0,
        shape * (S_total / safe_integrated),
        jnp.zeros_like(shape),
    )
  raise TypeError(f'Unknown profile runtime params: {type(profile_params)}')


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ParticleFeedbackSource(generic_particle_source.GenericParticleSource):
  """Generic particle source with PID feedback control."""

  @override
  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> ParticleFeedbackState:
    """Returns the initial PID controller state at `runtime_params.t`."""
    dtype = jax_utils.get_dtype()
    zero = jnp.array(0.0, dtype=dtype)
    return ParticleFeedbackState(
        integral_term=zero,
        previous_error=zero,
        smoothed_derivative=zero,
        t=jnp.asarray(runtime_params.t, dtype=dtype),
        initialized=jnp.array(False),
    )

  @override
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      solver_profiles: solver_profiles_lib.SolverProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[
      core_sources.SourceOutput,
      ParticleFeedbackState,
  ]:
    """Calculates generic particle source with PID feedback control.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry of the torus.
      solver_profiles: Core plasma profiles.
      source_state: PID controller state carried over from the previous
        timestep.

    Returns:
      A tuple containing:
        - SourceOutput, with the particle_source_e profile populated.
        - Updated ParticleFeedbackState for this source.
    """
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)
    assert isinstance(source_state, ParticleFeedbackState)

    match source_params.average_type:
      case AverageType.LINE:
        current_avg_n_e = math_utils.line_average(
            solver_profiles.kinetic.n_e.value, geo
        )
      case AverageType.VOLUME:
        current_avg_n_e = math_utils.volume_average(
            solver_profiles.kinetic.n_e.value, geo
        )
      case _ as unknown:
        raise ValueError(f'Unknown average type: {unknown}')

    dtype = jax_utils.get_dtype()
    t = runtime_params.t
    error = source_params.target_average_n_e - current_avg_n_e
    dt = t - source_state.t
    calc_deriv = (
        physics_formulas.calculate_time_derivative_with_exponential_smoothing
    )

    def _initialize_state() -> ParticleFeedbackState:
      # Record initial error without a false derivative spike from 0 -> error.
      initial_integral = jnp.clip(
          source_state.integral_term,
          source_params.min_integral,
          source_params.max_integral,
      )
      return ParticleFeedbackState(
          integral_term=initial_integral,
          previous_error=error,
          smoothed_derivative=jnp.array(0.0, dtype=dtype),
          t=t,
          initialized=jnp.array(True),
      )

    def _step_or_hold_state() -> ParticleFeedbackState:
      def _advance_state() -> ParticleFeedbackState:
        safe_dt = jnp.where(dt > 0.0, dt, 1.0)
        _, smoothed_derivative = calc_deriv(
            current_value=error,
            previous_value=source_state.previous_error,
            previous_smoothed_derivative=source_state.smoothed_derivative,
            dt=safe_dt,
            smoothing_time_scale=(
                source_params.derivative_smoothing_time_scale
            ),
        )
        integral_term = jnp.clip(
            source_state.integral_term
            + source_params.integral_gain * error * dt,
            source_params.min_integral,
            source_params.max_integral,
        )
        return ParticleFeedbackState(
            integral_term=integral_term,
            previous_error=error,
            smoothed_derivative=smoothed_derivative,
            t=t,
            initialized=jnp.array(True),
        )

      def _hold_state() -> ParticleFeedbackState:
        return ParticleFeedbackState(
            integral_term=jnp.clip(
                source_state.integral_term,
                source_params.min_integral,
                source_params.max_integral,
            ),
            previous_error=source_state.previous_error,
            smoothed_derivative=source_state.smoothed_derivative,
            t=source_state.t,
            initialized=source_state.initialized,
        )

      return jax.lax.cond(dt > 0.0, _advance_state, _hold_state)

    updated_state = jax.lax.cond(
        source_state.initialized,
        _step_or_hold_state,
        _initialize_state,
    )

    S_p = source_params.proportional_gain * error
    S_i = updated_state.integral_term
    S_d = source_params.derivative_gain * updated_state.smoothed_derivative
    S_feedback = S_p + S_i + S_d
    S_total = jnp.clip(source_params.S_feedforward + S_feedback, 0.0, jnp.inf)

    profile = _compute_radial_profile(source_params.profile, S_total, geo)
    return core_sources.SourceOutput(particle_source_e=profile), updated_state


class ParticleFeedbackSourceConfig(base.SourceConfigBase):
  """Config for the generic particle source with PID feedback control.

  The total particle injection rate is computed using a PID feedback control
  loop added to a baseline feedforward rate:

  ```
  S_total = max(0, S_feedforward + S_p + S_i + S_d)
  ```

  with density error `e(t) = target_average_n_e(t) - current_avg_n_e(t)` and:
    - `S_p = proportional_gain * e(t)`
    - `S_i = clip(S_i_prev + integral_gain * e(t) * dt, min_integral,
      max_integral)`
    - `S_d = derivative_gain * (de/dt)_smoothed`

  where `current_avg_n_e` is the instantaneous line- or volume-averaged electron
  density, and `(de/dt)_smoothed` is the finite-difference time derivative of
  `e(t)` optionally filtered with exponential smoothing time constant
  `derivative_smoothing_time_scale`. To prevent unphysical particle removal
  during over-fueling transients, the total source is clipped to be
  non-negative (>= 0).

  The resulting total particle injection rate [particles/s] is deposited
  radially using the shape configured by `profile` (`'exponential'`,
  `'gaussian'`, or `'prescribed'`).

  Attributes:
    model_name: Name of the generic particle model (`'feedback'`).
    profile: Radial deposition shape configuration, discriminated by
      `profile_type` (`'exponential'`, `'gaussian'`, or `'prescribed'`).
      Defaults to an inward edge exponential (`width=0.05`).
    S_feedforward: Baseline feedforward particle injection rate [particles/s].
    average_type: Method for spatial electron density averaging (`'line'` or
      `'volume'`) used to calculate the feedback error.
    proportional_gain: Proportional feedback gain [m^3 / s] mapping density
      error to an incremental particle fueling rate.
    integral_gain: Integral feedback gain [m^3 / s^2] mapping time-integrated
      density error to an incremental particle fueling rate.
    derivative_gain: Derivative feedback gain [m^3] mapping the rate of change
      of density error to an incremental particle fueling rate.
    derivative_smoothing_time_scale: Exponential smoothing time constant [s] for
      the error time derivative `de/dt`. `0.0` uses the raw finite difference.
    min_integral: Optional lower bound [particles/s] on the integral term `S_i`
      for anti-windup clamping. If `None`, unbounded below (`-inf`).
    max_integral: Optional upper bound [particles/s] on the integral term `S_i`
      for anti-windup clamping. If `None`, unbounded above (`+inf`).
    target_average_n_e: Desired target average electron density [m^-3] for
      feedback regulation.
  """

  model_name: Annotated[Literal['feedback'], torax_pydantic.JAX_STATIC] = (
      'feedback'
  )
  profile: ProfileConfig = pydantic.Field(
      default_factory=ExponentialProfileConfig
  )
  S_feedforward: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e22)
  )
  average_type: Annotated[AverageType, torax_pydantic.JAX_STATIC] = (
      AverageType.LINE
  )
  proportional_gain: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  integral_gain: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  derivative_gain: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  derivative_smoothing_time_scale: (
      torax_pydantic.NonNegativeTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(0.0)
  min_integral: torax_pydantic.TimeVaryingScalar | None = None
  max_integral: torax_pydantic.TimeVaryingScalar | None = None
  target_average_n_e: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.77e20)
  )

  @pydantic.model_validator(mode='after')
  def _validate_integral_bounds(self) -> Self:
    if self.min_integral is not None and self.max_integral is not None:
      check_times = np.union1d(self.min_integral.time, self.max_integral.time)
      min_vals = np.asarray(self.min_integral.get_value(check_times))
      max_vals = np.asarray(self.max_integral.get_value(check_times))
      if np.any(min_vals > max_vals):
        raise ValueError(
            'min_integral must be less than or equal to max_integral at all'
            ' times.'
        )
    return self

  @override
  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    """See base class."""
    base_params = super().build_runtime_params(t)
    dtype = jax_utils.get_dtype()
    min_integral = (
        self.min_integral.get_value(t)
        if self.min_integral is not None
        else jnp.array(-jnp.inf, dtype=dtype)
    )
    max_integral = (
        self.max_integral.get_value(t)
        if self.max_integral is not None
        else jnp.array(jnp.inf, dtype=dtype)
    )
    return RuntimeParams(
        **vars(base_params),
        profile=self.profile.build_runtime_params(t),
        S_feedforward=self.S_feedforward.get_value(t),
        target_average_n_e=self.target_average_n_e.get_value(t),
        average_type=self.average_type,
        proportional_gain=self.proportional_gain.get_value(t),
        integral_gain=self.integral_gain.get_value(t),
        derivative_gain=self.derivative_gain.get_value(t),
        derivative_smoothing_time_scale=(
            self.derivative_smoothing_time_scale.get_value(t)
        ),
        min_integral=min_integral,
        max_integral=max_integral,
    )

  @override
  def build_source(self) -> ParticleFeedbackSource:
    """See base class."""
    return ParticleFeedbackSource()
