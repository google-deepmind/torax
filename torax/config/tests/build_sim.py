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

"""Unit tests for torax.config.build_sim."""

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import build_sim


class BuildSimTest(parameterized.TestCase):
  """Unit tests for the `torax.config.build_sim` module."""

  def test_build_sim_raises_error_with_missing_keys(self):
    with self.assertRaises(ValueError):
      build_sim.build_sim_from_config({})

  def test_build_sim(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_sim_from_config({
          'runtime_params': {},
          'geometry': {},
          'sources': {},
          'transport': {},
          'stepper': {},
          'time_step_calculator': {},
      })

  def test_build_runtime_params_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_runtime_params_from_config({})

  def test_build_geometry_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_geometry_from_config({})

  def test_build_sources_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_sources_from_config({})

  def test_build_transport_model_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_transport_model_from_config({})

  def test_build_stepper_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_stepper_from_config({})

  def test_build_time_step_calculator_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_time_step_calculator_from_config({})


if __name__ == '__main__':
  absltest.main()
