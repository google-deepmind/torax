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

"""Unit tests for torax.transport_model.quasilinear_transport_model."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax import core_profile_setters
from torax import geometry
from torax import state
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


def _get_model_inputs(transport: quasilinear_transport_model.RuntimeParams):
  """Returns the model inputs for testing."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  geo = geometry.build_circular_geometry()
  source_models_builder = source_models_lib.SourceModelsBuilder()
  source_models = source_models_builder()
  dynamic_runtime_params_slice = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport,
          sources=source_models_builder.runtime_params,
          torax_mesh=geo.torax_mesh,
      )(
          t=runtime_params.numerics.t_initial,
      )
  )
  core_profiles = core_profile_setters.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      source_models=source_models,
  )
  return dynamic_runtime_params_slice, geo, core_profiles


class QuasilinearTransportModelTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.quasilinear_transport_model` module."""

  # pylint: disable=invalid-name

  def test_quasilinear_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    transport = quasilinear_transport_model.RuntimeParams()

    transport_model = FakeQuasilinearTransportModel()
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    core_transport = transport_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    expected_shape = geo.rho_face_norm.shape

    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='DVeff_False',
          DVeff=False,
          An_min=0.05,
          expected_zero_v_face_el=False,
          expected_zero_d_face_el=False,
      ),
      dict(
          testcase_name='DVeff_True_An_min_less_than_Ane',
          DVeff=True,
          An_min=0.05,
          expected_zero_v_face_el=True,
          expected_zero_d_face_el=False,
      ),
      dict(
          testcase_name='DVeff_True_An_min_greater_than_Ane',
          DVeff=True,
          An_min=2.0,
          expected_zero_v_face_el=False,
          expected_zero_d_face_el=True,
      ),
  )
  def test_quasilinear_transport_model_dveff(
      self, DVeff, An_min, expected_zero_v_face_el, expected_zero_d_face_el
  ):
    """Tests that the DVeff approach options behaves as expected."""
    transport = quasilinear_transport_model.RuntimeParams(
        DVeff=DVeff,
        An_min=An_min,
        **runtime_params_lib.RuntimeParams(Demin=0.0, Vemin=0.0)
    )
    transport_model = FakeQuasilinearTransportModel()
    core_transport = transport_model(*_get_model_inputs(transport))
    self.assertEqual(
        (jnp.sum(jnp.abs(core_transport.v_face_el)) == 0.0),
        expected_zero_v_face_el,
    )
    self.assertEqual(
        (jnp.sum(jnp.abs(core_transport.d_face_el)) == 0.0),
        expected_zero_d_face_el,
    )


class FakeQuasilinearTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Fake QuasilinearTransportModel for testing purposes."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    quasilinear_inputs = quasilinear_transport_model.QuasilinearInputs(
        chiGB=jnp.array(4.0),
        Rmin=jnp.array(0.5),
        Rmaj=jnp.array(1.0),
        Ati=jnp.array(1.1),
        Ate=jnp.array(1.2),
        Ane=jnp.array(1.3),
    )
    transport = dynamic_runtime_params_slice.transport
    # Assert required for pytype.
    assert isinstance(
        transport,
        quasilinear_transport_model.DynamicRuntimeParams,
    )
    return self._make_core_transport(
        qi=jnp.ones(geo.rho_face_norm.shape) * 0.4,
        qe=jnp.ones(geo.rho_face_norm.shape) * 0.5,
        pfe=jnp.ones(geo.rho_face_norm.shape) * 1.6,
        quasilinear_inputs=quasilinear_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


if __name__ == '__main__':
  absltest.main()
