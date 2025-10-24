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

"""Tests for PhasedTimeStepCalculator."""

import unittest
from unittest import mock
import jax.numpy as jnp
from torax._src import state as state_module
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.time_step_calculator import phased_time_step_calculator


class TestPhasedTimeStepCalculator(unittest.TestCase):
  """Test the PhasedTimeStepCalculator."""

  def setUp(self):
    self.calculator = phased_time_step_calculator.PhasedTimeStepCalculator()

    # Mock objects for testing
    self.geo = mock.MagicMock()
    self.core_transport = mock.MagicMock()

  def test_phased_dt_calculation(self):
    """Test that correct dt is returned based on current time."""
    # Set up phased time windows
    phased_dt_windows = ((0.5, 3.0), (3.0, 13.0), (13.0, 15.0))
    phased_dt_values = (0.2, 0.02, 0.1)

    numerics = numerics_lib.Numerics(
        t_initial=0.5,
        t_final=15.0,
        fixed_dt=1e-2,
        phased_dt_windows=phased_dt_windows,
        phased_dt_values=phased_dt_values,
    )

    # Test different time points
    test_cases = [
        (1.0, 0.2),   # First window: 0.5s to 3.0s, dt=0.2
        (5.0, 0.02),  # Second window: 3.0s to 13.0s, dt=0.02
        (14.0, 0.1),  # Third window: 13.0s to 15.0s, dt=0.1
    ]

    for current_time, expected_dt in test_cases:
      with self.subTest(time=current_time):
        # Create runtime params
        runtime_params = runtime_params_slice.RuntimeParams(
            numerics=numerics.build_runtime_params(current_time),
            transport=mock.MagicMock(),
            solver=mock.MagicMock(),
            sources={},
            plasma_composition=mock.MagicMock(),
            profile_conditions=mock.MagicMock(),
            neoclassical=mock.MagicMock(),
            pedestal=mock.MagicMock(),
            mhd=mock.MagicMock(),
            time_step_calculator=mock.MagicMock(),
        )

        # Create core profiles with specific time
        core_profiles = mock.MagicMock()
        core_profiles.t = current_time

        # Calculate dt
        dt = self.calculator._next_dt(
            runtime_params=runtime_params,
            geo=self.geo,
            core_profiles=core_profiles,
            core_transport=self.core_transport,
        )

        self.assertAlmostEqual(float(dt), expected_dt)

  def test_fallback_to_fixed_dt_when_no_phased_config(self):
    """Test fallback to fixed_dt when no phased configuration is provided."""
    numerics = numerics_lib.Numerics(
        fixed_dt=0.05,
        phased_dt_windows=None,
        phased_dt_values=None,
    )

    runtime_params = runtime_params_slice.RuntimeParams(
        numerics=numerics.build_runtime_params(1.0),
        transport=mock.MagicMock(),
        solver=mock.MagicMock(),
        sources={},
        plasma_composition=mock.MagicMock(),
        profile_conditions=mock.MagicMock(),
        neoclassical=mock.MagicMock(),
        pedestal=mock.MagicMock(),
        mhd=mock.MagicMock(),
        time_step_calculator=mock.MagicMock(),
    )

    core_profiles = mock.MagicMock()
    core_profiles.t = 1.0

    dt = self.calculator._next_dt(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=core_profiles,
        core_transport=self.core_transport,
    )

    self.assertAlmostEqual(float(dt), 0.05)

  def test_equality_and_hash(self):
    """Test equality and hash methods."""
    calc1 = phased_time_step_calculator.PhasedTimeStepCalculator()
    calc2 = phased_time_step_calculator.PhasedTimeStepCalculator()

    self.assertEqual(calc1, calc2)
    self.assertEqual(hash(calc1), hash(calc2))


if __name__ == '__main__':
  unittest.main()