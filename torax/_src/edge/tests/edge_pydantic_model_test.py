# Copyright 2025 DeepMind Technologies Limited
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

from absl.testing import absltest
from torax._src.edge import pydantic_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class ExtendedLengyelPydanticModelTest(absltest.TestCase):

  def test_extended_lengyel_defaults(self):
    """Checks default values for the extended lengyel config."""
    config = pydantic_model.ExtendedLengyelConfig()
    self.assertEqual(config.model_name, 'extended_lengyel')
    self.assertEqual(
        config.computation_mode, pydantic_model.ComputationMode.FORWARD
    )
    self.assertEqual(config.solver_mode, pydantic_model.SolverMode.HYBRID)

  def test_torax_config_integration(self):
    """Ensures ToraxConfig can parse the new edge field."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
    }

    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    self.assertIsNotNone(torax_config.edge)
    self.assertIsInstance(
        torax_config.edge, pydantic_model.ExtendedLengyelConfig
    )
    # Extra assert to make pytype happy.
    assert isinstance(torax_config.edge, pydantic_model.ExtendedLengyelConfig)
    self.assertEqual(
        torax_config.edge.computation_mode,
        pydantic_model.ComputationMode.INVERSE,
    )

  def test_torax_config_no_edge(self):
    """Ensures ToraxConfig works fine without an edge config."""
    config_dict = default_configs.get_default_config_dict()
    # No 'edge' key in config_dict

    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertIsNone(torax_config.edge)


if __name__ == '__main__':
  absltest.main()
