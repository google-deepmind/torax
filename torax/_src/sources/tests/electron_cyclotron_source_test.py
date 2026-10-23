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
from absl.testing import absltest
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import electron_cyclotron_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import torax_config as torax_config_lib


def _build_torax_config(
    power: float, current_drive_efficiency: float
) -> torax_config_lib.ToraxConfig:
  """Returns a config with a Gaussian EC source of uniform efficiency."""
  config = default_configs.get_default_config_dict()
  config["sources"] = {
      electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID: {
          "P_total": power,
          "gaussian_location": 0.3,
          "current_drive_efficiency": {
              0.0: {
                  0.0: current_drive_efficiency,
                  1.0: current_drive_efficiency,
              }
          },
      }
  }
  return torax_config_lib.ToraxConfig.from_dict(config)


def _build_runtime_params(torax_config: torax_config_lib.ToraxConfig):
  return build_runtime_params.RuntimeParamsProvider.from_config(torax_config)(
      t=torax_config.numerics.t_initial
  )


def _build_plasma_state(
    torax_config: torax_config_lib.ToraxConfig, runtime_params
):
  """Returns the geometry and initial core profiles for the config."""
  geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
  core_profiles = initialization.initial_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      source_models=torax_config.sources.build_models(),
      neoclassical_model=torax_config.neoclassical.build_model(),
  )
  return geo, core_profiles


def _calc_ec_current(runtime_params, geo, core_profiles) -> np.ndarray:
  source = electron_cyclotron_source.ElectronCyclotronSource()
  _, j_parallel = source.get_value(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=None,
      conductivity=None,
  )
  return np.asarray(j_parallel)


class ElectronCyclotronSourceTest(test_lib.MultipleProfileSourceTestCase):
  """Tests for ElectronCyclotronSource."""

  source_config_class = electron_cyclotron_source.ElectronCyclotronSourceConfig
  source_name = electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID

  def test_negative_current_drive_efficiency_reverses_current(self):
    positive_config = _build_torax_config(
        power=10e6, current_drive_efficiency=0.25
    )
    negative_config = _build_torax_config(
        power=10e6, current_drive_efficiency=-0.25
    )
    positive_runtime_params = _build_runtime_params(positive_config)
    negative_runtime_params = _build_runtime_params(negative_config)
    # Evaluate both efficiencies on a single plasma state: the initial psi
    # includes the EC-driven current, so building the state separately for each
    # efficiency would change q and therefore the <j.B> geometric factor.
    geo, core_profiles = _build_plasma_state(
        positive_config, positive_runtime_params
    )

    positive_current = _calc_ec_current(
        positive_runtime_params, geo, core_profiles
    )
    negative_current = _calc_ec_current(
        negative_runtime_params, geo, core_profiles
    )

    self.assertTrue(np.any(positive_current > 0.0))
    self.assertTrue(np.all(np.isfinite(negative_current)))
    np.testing.assert_allclose(negative_current, -positive_current)

  def test_zero_power_gives_zero_current(self):
    torax_config = _build_torax_config(power=0.0, current_drive_efficiency=0.25)
    runtime_params = _build_runtime_params(torax_config)
    geo, core_profiles = _build_plasma_state(torax_config, runtime_params)

    j_parallel = _calc_ec_current(runtime_params, geo, core_profiles)

    np.testing.assert_array_equal(j_parallel, 0.0)


if __name__ == "__main__":
  absltest.main()
