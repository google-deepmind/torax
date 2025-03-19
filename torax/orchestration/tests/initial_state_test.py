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
from unittest import mock

from absl.testing import absltest
from torax import sim
from torax.config import build_runtime_params
from torax.orchestration import initial_state
from torax.orchestration import step_function
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import sim_test_case
from torax.torax_pydantic import model_config


class InitialStateTest(sim_test_case.SimTestCase):

  def test_from_file_restart(self):
    restart_config = 'test_iterhybrid_rampup_restart.py'

    config_module = self._get_config_module(restart_config)
    torax_config = model_config.ToraxConfig.from_dict(config_module.CONFIG)

    stepper = mock.MagicMock()
    stepper.source_models = source_models_lib.SourceModels(
        torax_config.sources.source_model_config
    )
    step_fn = mock.create_autospec(step_function.SimulationStepFn,
                                   stepper=stepper)

    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=torax_config.runtime_params,
            sources=torax_config.sources,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
            stepper=torax_config.stepper,
        )
    )
    dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=torax_config.runtime_params,
            pedestal=torax_config.pedestal,
            transport=torax_config.transport,
            sources=torax_config.sources,
            stepper=torax_config.stepper,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
        )
    )
    dynamic_runtime_params_slice_for_init, geo_for_init = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.runtime_params.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )

    non_restart = sim.get_initial_state(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
        geo=geo_for_init,
        step_fn=step_fn,
    )

    result = initial_state.initial_state_from_file_restart(
        file_restart=torax_config.restart,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_for_init=dynamic_runtime_params_slice_for_init,
        geo_for_init=geo_for_init,
        step_fn=step_fn,
    )

    self.assertNotEqual(result.post_processed_outputs.E_cumulative_fusion,
                        0.)
    self.assertNotEqual(result.post_processed_outputs.E_cumulative_external,
                        0.)

    self.assertNotEqual(result.core_profiles, non_restart.core_profiles)
    self.assertNotEqual(result.t, non_restart.t)
    assert torax_config.restart is not None
    self.assertEqual(result.t, torax_config.restart.time)


if __name__ == '__main__':
  absltest.main()
