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
from torax.config import runtime_params as runtime_params_lib


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

  def test_build_runtime_params_from_empty_config(self):
    """An empty config should return all defaults."""
    runtime_params = build_sim.build_runtime_params_from_config({})
    defaults = runtime_params_lib.GeneralRuntimeParams()
    self.assertEqual(runtime_params, defaults)

  def test_build_runtime_params_raises_error_with_incorrect_args(self):
    """If an incorrect key is provided, an error should be raised."""
    with self.assertRaises(KeyError):
      build_sim.build_runtime_params_from_config({'incorrect_key': 'value'})

  def test_general_runtime_params_with_time_dependent_args(self):
    """Tests that we can build all types of attributes in the runtime params."""
    runtime_params = build_sim.build_runtime_params_from_config({
        'plasma_composition': {
            'Ai': 0.1,  # scalar fields.
            'Zeff': {0: 0.1, 1: 0.2, 2: 0.3},  # time-dependent.
        },
        'profile_conditions': {
            'nbar_is_fGW': False,  # scalar fields.
            'Ip': {0: 0.2, 1: 0.4, 2: 0.6},  # time-dependent.
        },
        'numerics': {
            'q_correction_factor': 0.2,  # scalar fields.
            'resistivity_mult': {0: 0.3, 1: 0.6, 2: 0.9},  # time-dependent.
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(runtime_params.plasma_composition.Ai, 0.1)
    self.assertEqual(runtime_params.plasma_composition.Zeff[0], 0.1)
    self.assertEqual(runtime_params.profile_conditions.nbar_is_fGW, False)
    self.assertEqual(runtime_params.profile_conditions.Ip[1], 0.4)
    self.assertEqual(runtime_params.numerics.q_correction_factor, 0.2)
    self.assertEqual(runtime_params.numerics.resistivity_mult[2], 0.9)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')

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
