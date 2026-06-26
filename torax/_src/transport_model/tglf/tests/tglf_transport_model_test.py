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

"""Tests for the TGLF transport model."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model.tglf import tglf2py
# Internal import.


@absltest.skipIf(
    tglf2py.tglf2py_lib is None,
    "TGLF extension module 'tglf2py_lib' is not compiled/installed.",
)
class TGLFTransportModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("with_jit", True),
      ("without_jit", False),
  )
  def test_call(self, jit: bool):
    """Tests that the model can be called and executes real TGLF calculations."""
    config = default_configs.get_default_config_dict()
    config["transport"] = {"model_name": "tglf"}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    transport_model = torax_config.transport.build_transport_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    model_call = (
        jax.jit(transport_model.__call__) if jit else transport_model.__call__
    )
    outputs = model_call(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output_lib.PedestalModelOutput(
            rho_norm_ped_top=np.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
    )
    self.assertIsNotNone(outputs.chi_face_ion)
    self.assertIsNotNone(outputs.chi_face_el)

  def test_deprecated_params_warn_and_run(self):
    """Tests that deprecated config params raise warnings and the execution still runs."""
    config = default_configs.get_default_config_dict()
    config["transport"] = {
        "model_name": "tglf",
        "tglf_exec_path": "/deprecated/path",
        "output_directory": "/deprecated/dir",
        "verbose": True,
        "sat_rule": 0,  # legacy config param format
    }
    with self.assertLogs(level="WARNING") as log_watcher:
      torax_config = model_config.ToraxConfig.from_dict(config)
    log_output = "\n".join(log_watcher.output)
    self.assertIn("Config option 'tglf_exec_path' is deprecated", log_output)
    self.assertIn("Config option 'output_directory' is deprecated", log_output)
    self.assertIn("Config option 'verbose' is deprecated", log_output)
    self.assertIn(
        "Parsing TGLF settings from the legacy config kwargs is deprecated",
        log_output,
    )

    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    transport_model = torax_config.transport.build_transport_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    outputs = transport_model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output_lib.PedestalModelOutput(
            rho_norm_ped_top=np.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
    )
    self.assertIsNotNone(outputs.chi_face_ion)
    self.assertIsNotNone(outputs.chi_face_el)


if __name__ == "__main__":
  absltest.main()
