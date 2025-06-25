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
import pydantic
from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.qlknn.models import registry

from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import bohm_gyrobohm
from torax._src.transport_model import constant
from torax._src.transport_model import critical_gradient
from torax._src.transport_model import \
    pydantic_model as transport_pydantic_model
from torax._src.transport_model import qlknn_10d
from torax._src.transport_model import qlknn_transport_model
from torax._src.transport_model import runtime_params
from torax._src.transport_model import transport_model as transport_model_lib


class PydanticModelTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          pydantic_model=transport_pydantic_model.QLKNNTransportModel,
          expected_dynamic_params=qlknn_transport_model.DynamicRuntimeParams,
          expected_transport_model=qlknn_transport_model.QLKNNTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.ConstantTransportModel,
          expected_dynamic_params=constant.DynamicRuntimeParams,
          expected_transport_model=constant.ConstantTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.CriticalGradientTransportModel,
          expected_dynamic_params=critical_gradient.DynamicRuntimeParams,
          expected_transport_model=critical_gradient.CriticalGradientTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.BohmGyroBohmTransportModel,
          expected_dynamic_params=bohm_gyrobohm.DynamicRuntimeParams,
          expected_transport_model=bohm_gyrobohm.BohmGyroBohmTransportModel,
      ),
  )
  def test_build_transport_model(
      self,
      pydantic_model: type[transport_pydantic_model.TransportConfig],
      expected_dynamic_params: type[runtime_params.DynamicRuntimeParams],
      expected_transport_model: type[transport_model_lib.TransportModel],
  ):
    transport = pydantic_model()

    # Any TimeVaryingArrays in the transport model must have a grid set, so we
    # need to define a geometry. We use a circular geometry for convenience.
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(transport, geo.torax_mesh)

    dynamic_runtime_params = transport.build_dynamic_params(t=0.0)
    self.assertIsInstance(dynamic_runtime_params, expected_dynamic_params)
    transport_model = transport.build_transport_model()
    self.assertIsInstance(transport_model, expected_transport_model)

  def test_build_qualikiz_transport_model(self):
    try:
      # pylint: disable=g-import-not-at-top
      from torax._src.transport_model import qualikiz_transport_model

      # pylint: enable=g-import-not-at-top
    except ImportError:
      self.skipTest('Qualikiz transport model is not available.')

    transport = qualikiz_transport_model.QualikizTransportModelConfig()
    dynamic_runtime_params = transport.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_runtime_params, qualikiz_transport_model.DynamicRuntimeParams
    )
    transport_model = transport.build_transport_model()
    self.assertIsInstance(
        transport_model, qualikiz_transport_model.QualikizTransportModel
    )

  def test_qlknn_defaults(self):
    """Tests that correct default values are set for QLKNN."""
    transport = transport_pydantic_model.QLKNNTransportModel()
    self.assertIsInstance(
        transport, transport_pydantic_model.QLKNNTransportModel
    )
    self.assertEqual(transport.smoothing_width, 0.1)
    self.assertEqual(transport.ETG_correction_factor, 1.0 / 3.0)

  @parameterized.parameters(
      dict(
          model_name='',
          model_path='',
          expected_model_name=registry.DEFAULT_MODEL_NAME,
      ),
      dict(
          model_name='foo',
          model_path='',
          expected_model_name='foo',
      ),
      dict(
          model_name='foo',
          model_path='/path/to/model.qlknn',
          expected_model_name='foo',
      ),
      dict(
          model_name='',
          model_path='/path/to/model.qlknn',
          expected_model_name='',
      ),
      dict(
          model_name='',
          model_path='/path/to/qlknn-hyper-directory',
          expected_model_name=qlknn_10d.QLKNN10D_NAME,
      ),
  )
  def test_qlknn_model_name(self, model_name, model_path, expected_model_name):
    """Tests that the model name is resolved correctly."""
    transport = transport_pydantic_model.QLKNNTransportModel(
        qlknn_model_name=model_name,
        model_path=model_path,
    )
    self.assertEqual(transport.qlknn_model_name, expected_model_name)

  @parameterized.parameters(
      {'model_name': qlknn_10d.QLKNN10D_NAME, 'model_path': ''},
      {
          'model_name': qlknn_10d.QLKNN10D_NAME,
          'model_path': '/path/to/qlknn_7_11.qlknn',
      },
  )
  def test_qlknn_model_errors(self, model_name, model_path):
    with self.assertRaises(ValueError):
      transport_pydantic_model.QLKNNTransportModel(
          qlknn_model_name=model_name,
          model_path=model_path,
      )

  @parameterized.named_parameters(
      (
          'mixed_modes_inner_outer_fails',
          {'rho_inner': 0.3, 'rho_outer': ({0: 0.9}, 'step')},
          'rho_outer and rho_inner must have the same interpolation mode.',
          True,
      ),
      (
          'mixed_modes_min_max_fails',
          {'rho_min': 0.0, 'rho_max': ({0: 1.0}, 'step')},
          'rho_max and rho_min must have the same interpolation mode.',
          True,
      ),
      (
          'inner_greater_than_outer_fails',
          {'rho_inner': 0.9, 'rho_outer': 0.3},
          'rho_outer must be greater than rho_inner for all time.',
          True,
      ),
      (
          'min_greater_than_max_fails',
          {'rho_min': 0.9, 'rho_max': 0.3},
          'rho_max must be greater than rho_min for all time.',
          True,
      ),
      (
          'time_varying_inner_gt_outer_fails',
          {
              'rho_inner': {0: 0.3, 1: 0.95},
              'rho_outer': {0: 0.9, 1: 0.9},
          },
          'rho_outer must be greater than rho_inner for all time.',
          True,
      ),
      (
          'time_varying_min_gt_max_fails',
          {
              'rho_min': {0: 0.0, 1: 0.95},
              'rho_max': {0: 1.0, 1: 0.9},
          },
          'rho_max must be greater than rho_min for all time.',
          True,
      ),
      (
          'time_varying_linear_succeeds',
          {
              'rho_inner': {0: 0.2, 1: 0.3},
              'rho_outer': {0: 0.8, 1: 0.9},
              'rho_min': {0: 0.0, 1: 0.1},
              'rho_max': {0: 1.0, 1: 0.95},
          },
          None,
          False,
      ),
      (
          'time_varying_step_succeeds',
          {
              'rho_inner': ({0: 0.2, 1: 0.3}, 'step'),
              'rho_outer': ({0: 0.8, 1: 0.9}, 'step'),
              'rho_min': ({0: 0.0, 1: 0.1}, 'step'),
              'rho_max': ({0: 1.0, 1: 0.95}, 'step'),
          },
          None,
          False,
      ),
  )
  def test_rho_validation(self, params, error_msg, should_fail):
    if should_fail:
      with self.assertRaisesRegex(pydantic.ValidationError, error_msg):
        transport_pydantic_model.ConstantTransportModel(**params)
    else:
      transport_pydantic_model.ConstantTransportModel(**params)


if __name__ == '__main__':
  absltest.main()
