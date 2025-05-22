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
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.config import numerics
from torax._src.config import plasma_composition
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import initialization
from torax._src.sources import source_models as source_models_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import combined
from torax._src.pedestal_model import pedestal_model


def _get_model_and_inputs_from_config(config):
  torax_config = model_config.ToraxConfig.from_dict(config)
  model = torax_config.transport.build_transport_model()
  geo = torax_config.geometry.build_provider(t=torax_config.numerics.t_initial)
  dynamic_runtime_params_slice = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
          torax_config
      )(
          t=torax_config.numerics.t_initial,
      )
  )
  static_runtime_params_slice = (
      build_runtime_params.build_static_params_from_config(torax_config)
  )
  core_profiles = initialization.initial_core_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      source_models_lib.SourceModels(
          sources=torax_config.sources, neoclassical=torax_config.neoclassical
      ),
  )
  mock_pedestal_outputs = mock.create_autospec(
      pedestal_model.PedestalModelOutput,
      instance=True,
      rho_norm_ped_top=0.91,
  )
  return (
      model,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      mock_pedestal_outputs,
  )


# pylint: disable=invalid-name
class CombinedTransportModelTest(absltest.TestCase):

  def test_call_implementation(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'rho_max': 0.5, 'chi_i': 2.0},
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 1.0},
        ],
    }
    (
        model,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        mock_pedestal_outputs,
    ) = _get_model_and_inputs_from_config(config)
    transport_coeffs = model._call_implementation(
        dynamic_runtime_params_slice, geo, core_profiles, mock_pedestal_outputs
    )
    target = jnp.where(geo.rho_face_norm <= 0.91, 1.0, 0.0)
    target = jnp.where(geo.rho_face_norm == 0.5, 3.0, target)
    target = jnp.where(geo.rho_face_norm < 0.5, 2.0, target)
    np.testing.assert_allclose(transport_coeffs.chi_face_ion, target)

  def test_error_if_patches_set_on_children(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'apply_inner_patch': True},
            {'model_name': 'constant'},
        ],
    }
    with self.assertRaises(ValueError):
      _get_model_and_inputs_from_config(config)


if __name__ == '__main__':
  absltest.main()
