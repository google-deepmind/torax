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

"""Tests for the pellet-aware time step calculator.

The calculator only reads the current time and the 'pellet' source runtime
parameters, so '_next_dt' is exercised directly against mocked inputs with a
real FixedTimeStepCalculator as the base (away-from-pellet) calculator. The
pellet source is read by duck typing (getattr with defaults) because its
concrete runtime params class lives in the pellet source model, not in TORAX.
The pellet mock therefore only exposes the declared attributes, so an omitted
one exercises the calculator's default.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import pellet_aware_time_step_calculator


_ABLATION = 1e-3


def _pellet(**attrs):
  """A fake pellet source runtime params exposing only the given attributes."""
  return mock.Mock(spec=list(attrs), **attrs)


def _runtime_params(pellet, dt_standard):
  """Runtime params with a 'pellet' source and a fixed base step."""
  numerics = mock.create_autospec(
      numerics_lib.RuntimeParams, instance=True, fixed_dt=dt_standard
  )
  return mock.create_autospec(
      runtime_params_lib.RuntimeParams,
      instance=True,
      sources={'pellet': pellet},
      numerics=numerics,
  )


def _sim_state(t, geometry=None, core_profiles=None):
  return mock.create_autospec(
      sim_state_lib.SimState,
      instance=True,
      t=t,
      geometry=geometry,
      core_profiles=core_profiles,
  )


def _calculator(**kwargs):
  """A pellet-aware calculator with a fixed base and given overrides."""
  kwargs.setdefault('trigger_tolerance', 1e-8)
  return pellet_aware_time_step_calculator.PelletAwareTimeStepCalculator(
      base_calculator=fixed_time_step_calculator.FixedTimeStepCalculator(),
      **kwargs,
  )


class PelletAwareTimeStepCalculatorTest(parameterized.TestCase):

  def _dt(self, pellet, t, dt_standard=0.1, calculator=None):
    calculator = calculator or _calculator()
    return float(
        calculator._next_dt(
            _runtime_params(pellet, dt_standard), _sim_state(t)
        )
    )

  def _assert_dt(self, got, want):
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)

  @parameterized.named_parameters(
      # Far from any trigger: the base step is used.
      dict(testcase_name='far_before', t=0.0, want=0.1),
      # Between triggers: the base step is used (next trigger is far).
      dict(testcase_name='between', t=3.0, want=0.1),
      # Approaching a trigger: dt shrinks to land exactly on it.
      dict(testcase_name='shrink_to_first', t=1.95, want=0.05),
      dict(testcase_name='shrink_to_second', t=5.95, want=0.05),
      # At a trigger: the whole ablation window is a single step.
      dict(testcase_name='at_first', t=2.0, want=_ABLATION),
      dict(testcase_name='at_second', t=6.0, want=_ABLATION),
  )
  def test_trigger_times_alignment(self, t, want):
    pellet = _pellet(trigger_times=(2.0, 6.0), ablation_time=_ABLATION)
    self._assert_dt(self._dt(pellet, t), want)

  def test_ablation_window_is_never_split(self):
    # Even when the base step is smaller than the ablation window, the trigger
    # step covers the whole window in one step.
    pellet = _pellet(trigger_times=(2.0,), ablation_time=_ABLATION)
    self._assert_dt(self._dt(pellet, t=2.0, dt_standard=1e-4), _ABLATION)

  @parameterized.named_parameters(
      # First pellet fires exactly at frequency_t_start.
      dict(testcase_name='at_start', t=1.0, want=_ABLATION),
      # Mid-period: the base step is used.
      dict(testcase_name='mid_period', t=1.25, want=0.1),
      # Approaching a boundary: dt shrinks to land exactly on it.
      dict(testcase_name='shrink_to_boundary', t=1.45, want=0.05),
      # Subsequent boundaries fire too.
      dict(testcase_name='at_second_boundary', t=1.5, want=_ABLATION),
      dict(testcase_name='at_third_boundary', t=2.0, want=_ABLATION),
  )
  def test_frequency_alignment(self, t, want):
    pellet = _pellet(
        frequency=2.0,  # period 0.5 s
        frequency_t_start=1.0,
        injection_enabled=True,
        ablation_time=_ABLATION,
    )
    self._assert_dt(self._dt(pellet, t), want)

  def test_frequency_defaults_start_to_zero(self):
    # Without frequency_t_start the phase is measured from t=0, so a boundary
    # sits at t=0.
    pellet = _pellet(frequency=2.0, ablation_time=_ABLATION)
    self._assert_dt(self._dt(pellet, t=0.0), _ABLATION)

  def test_frequency_wrap_detects_near_boundary(self):
    # A step landing a hair before a boundary (within tolerance) still fires:
    # the phase wraps to 0 instead of staying just below the period. This guards
    # the float robustness of the boundary detection.
    pellet = _pellet(
        frequency=2.0,  # period 0.5 s
        frequency_t_start=1.0,
        trigger_tolerance=1e-3,
        ablation_time=_ABLATION,
    )
    self._assert_dt(self._dt(pellet, t=1.5 - 1e-5), _ABLATION)

  def test_injection_disabled_uses_base_step(self):
    # When the injector is off, no pellet fires and the base step is used even
    # on a period boundary.
    pellet = _pellet(
        frequency=2.0,
        frequency_t_start=1.0,
        injection_enabled=False,
        ablation_time=_ABLATION,
    )
    self._assert_dt(self._dt(pellet, t=1.0), 0.1)

  def test_injection_enabled_defaults_on(self):
    # A source that does not expose injection_enabled keeps firing.
    pellet = _pellet(
        frequency=2.0, frequency_t_start=1.0, ablation_time=_ABLATION
    )
    self._assert_dt(self._dt(pellet, t=1.0), _ABLATION)

  def test_uses_source_trigger_tolerance(self):
    # The pellet source's own trigger_tolerance takes precedence over the
    # calculator's, so the step alignment agrees with the source's deposition.
    pellet = _pellet(
        trigger_times=(2.0,), ablation_time=_ABLATION, trigger_tolerance=1e-3
    )
    # 5e-4 away from the trigger: within the source tolerance (1e-3) but well
    # outside the calculator default (1e-8), so the pellet fires.
    self._assert_dt(self._dt(pellet, t=2.0005), _ABLATION)

  def test_model_ablation_time_hook(self):
    # A source exposing use_model_ablation_time and an ablation_step method sets
    # the window from the model output, not the constant ablation_time.
    received = {}

    def ablation_step(geometry, core_profiles):
      received['geometry'] = geometry
      received['core_profiles'] = core_profiles
      return None, 5e-3  # (deposited profile, ablation time)

    pellet = _pellet(
        trigger_times=(2.0,),
        ablation_time=_ABLATION,
        use_model_ablation_time=True,
        ablation_step=ablation_step,
    )
    dt = float(
        _calculator()._next_dt(
            _runtime_params(pellet, 0.1),
            _sim_state(t=2.0, geometry='geo', core_profiles='core'),
        )
    )
    self._assert_dt(dt, 5e-3)
    self.assertEqual(received['geometry'], 'geo')
    self.assertEqual(received['core_profiles'], 'core')

  @parameterized.named_parameters(
      # Inside the post-pellet window: dt_after_pellet is used.
      dict(testcase_name='inside_window', t=2.02, want=0.005),
      # Near the window end: dt shrinks to land exactly on it.
      dict(testcase_name='near_window_end', t=2.048, want=0.002),
      # Past the window: the base step is used again.
      dict(testcase_name='after_window', t=2.1, want=0.1),
  )
  def test_post_pellet_window_trigger_times(self, t, want):
    pellet = _pellet(trigger_times=(2.0,), ablation_time=_ABLATION)
    calculator = _calculator(window_after_pellet=0.05, dt_after_pellet=0.005)
    self._assert_dt(self._dt(pellet, t, calculator=calculator), want)

  def test_post_pellet_window_frequency(self):
    # The post-pellet window also applies to the frequency mode, measured from
    # the period boundary.
    pellet = _pellet(
        frequency=2.0, frequency_t_start=1.0, ablation_time=_ABLATION
    )
    calculator = _calculator(window_after_pellet=0.05, dt_after_pellet=0.005)
    self._assert_dt(self._dt(pellet, t=1.02, calculator=calculator), 0.005)

  @parameterized.named_parameters(
      dict(
          testcase_name='both',
          pellet=dict(
              trigger_times=(2.0,), frequency=2.0, ablation_time=_ABLATION
          ),
      ),
      dict(testcase_name='neither', pellet=dict(ablation_time=_ABLATION)),
  )
  def test_requires_exactly_one_of_trigger_times_or_frequency(self, pellet):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      self._dt(_pellet(**pellet), t=0.0)

  def test_equality_and_hash(self):
    # Equality and hashing (needed for JAX static arguments) cover the base
    # calculator and every alignment parameter.
    fixed = fixed_time_step_calculator.FixedTimeStepCalculator
    make = pellet_aware_time_step_calculator.PelletAwareTimeStepCalculator
    calculator = make(fixed(), trigger_tolerance=1e-8)
    self.assertEqual(calculator, make(fixed(), trigger_tolerance=1e-8))
    self.assertEqual(
        hash(calculator), hash(make(fixed(), trigger_tolerance=1e-8))
    )
    # Differing in the base calculator or any single parameter breaks equality.
    self.assertNotEqual(
        calculator, make(chi_time_step_calculator.ChiTimeStepCalculator())
    )
    self.assertNotEqual(calculator, make(fixed(), trigger_tolerance=1e-6))
    self.assertNotEqual(calculator, make(fixed(), window_after_pellet=0.1))
    self.assertNotEqual(calculator, make(fixed(), dt_after_pellet=0.01))


if __name__ == '__main__':
  absltest.main()