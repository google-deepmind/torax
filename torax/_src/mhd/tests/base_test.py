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
import os

from absl.testing import absltest
from torax._src.config import config_loader
from torax._src.orchestration import run_simulation
from torax._src.test_utils import paths


class BaseTest(absltest.TestCase):

  def test_different_mhd_models_have_same_hash_and_equals(self):
    test_data_dir = paths.test_data_dir()
    torax_config = config_loader.build_torax_config_from_file(
        os.path.join(test_data_dir, "test_iterhybrid_rampup.py")
    )
    static_runtime_params_slice, _, _, _, _, _, step_fn = (
        run_simulation.prepare_simulation(torax_config)
    )
    model1 = torax_config.mhd.build_mhd_models(
        static_runtime_params_slice,
        step_fn.solver.transport_model,
        step_fn.solver.source_models,
        step_fn.solver.pedestal_model,
        step_fn.solver.neoclassical_models,
    )
    model2 = torax_config.mhd.build_mhd_models(
        static_runtime_params_slice,
        step_fn.solver.transport_model,
        step_fn.solver.source_models,
        step_fn.solver.pedestal_model,
        step_fn.solver.neoclassical_models,
    )
    self.assertEqual(model1, model2)
    self.assertEqual(hash(model1), hash(model2))


if __name__ == "__main__":
  absltest.main()
