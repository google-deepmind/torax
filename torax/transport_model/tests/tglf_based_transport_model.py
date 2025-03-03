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

"""Unit tests for torax.transport_model.tglf_based_transport_model."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from torax import core_profile_setters
from torax import state
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import set_tped_nped
from torax.sources import source_models as source_models_lib
from torax.transport_model import tglf_based_transport_model
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


def _get_model_inputs(transport: tglf_based_transport_model.RuntimeParams):
  """Returns the model inputs for testing."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  geo = circular_geometry.build_circular_geometry()
  source_models_builder = source_models_lib.SourceModelsBuilder()
  source_models = source_models_builder()
  pedestal_model_builder = (
      set_tped_nped.SetTemperatureDensityPedestalModelBuilder()
  )
  dynamic_runtime_params_slice = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport,
          sources=source_models_builder.runtime_params,
          pedestal=pedestal_model_builder.runtime_params,
          torax_mesh=geo.torax_mesh,
      )(
          t=runtime_params.numerics.t_initial,
      )
  )
  static_slice = runtime_params_slice.build_static_runtime_params_slice(
      runtime_params=runtime_params,
      source_runtime_params=source_models_builder.runtime_params,
      torax_mesh=geo.torax_mesh,
  )
  core_profiles = core_profile_setters.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_slice,
      geo=geo,
      source_models=source_models,
  )
  return dynamic_runtime_params_slice, geo, core_profiles


class TGLFBasedTransportModelTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.tglf_based_transport_model` module."""

  def test_tglf_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    transport = tglf_based_transport_model.RuntimeParams(
        **runtime_params_lib.RuntimeParams()
    )
    transport_model = FakeTGLFBasedTransportModel()
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    pedestal_model = set_tped_nped.SetTemperatureDensityPedestalModel()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )

    core_transport = transport_model(
        dynamic_runtime_params_slice, geo, core_profiles, pedestal_model_outputs
    )
    expected_shape = geo.rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  def test_tglf_based_transport_model_prepare_tglf_inputs_shapes(self):
    """Tests that the tglf inputs have the expected shapes."""
    transport = tglf_based_transport_model.RuntimeParams(
        **runtime_params_lib.RuntimeParams()
    )
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    transport_model = FakeTGLFBasedTransportModel()
    tglf_inputs = transport_model._prepare_tglf_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        geo=geo,
        core_profiles=core_profiles,
    )

    # Inputs that are 1D
    vector_keys = [
        'chiGB',
        'lref_over_lti',
        'lref_over_lte',
        'lref_over_lne',
        'lref_over_lni0',
        'lref_over_lni1',
        'Ti_over_Te',
        'drmaj',
        'q',
        's_hat',
        'nu_ee',
        'kappa',
        'kappa_shear',
        'delta',
        'delta_shear',
        'beta_e',
        'Zeff',
    ]
    # Inputs that are 0D
    scalar_keys = ['Rmaj', 'Rmin']

    expected_vector_length = geo.rho_face_norm.shape[0]
    for key in vector_keys:
      try:
        self.assertEqual(
            getattr(tglf_inputs, key).shape, (expected_vector_length,)
        )
      except Exception as e:
        print(key, getattr(tglf_inputs, key))
        raise e
    for key in scalar_keys:
      self.assertEqual(getattr(tglf_inputs, key).shape, ())


class FakeTGLFBasedTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Fake TGLFBasedTransportModel for testing purposes."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  # pylint: disable=invalid-name
  def prepare_tglf_inputs(
      self,
      Zeff_face: chex.Array,
      q_correction_factor: chex.Numeric,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tglf_based_transport_model.TGLFInputs:
    """Exposing prepare_tglf_inputs for testing."""
    return self._prepare_tglf_inputs(
        Zeff_face=Zeff_face,
        q_correction_factor=q_correction_factor,
        geo=geo,
        core_profiles=core_profiles,
    )

  # pylint: enable=invalid-name

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    tglf_inputs = self._prepare_tglf_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        geo=geo,
        core_profiles=core_profiles,
    )

    transport = dynamic_runtime_params_slice.transport
    # Assert required for pytype.
    assert isinstance(
        transport,
        tglf_based_transport_model.DynamicRuntimeParams,
    )

    return self._make_core_transport(
        qi=jnp.ones(geo.rho_face_norm.shape) * 0.4,
        qe=jnp.ones(geo.rho_face_norm.shape) * 0.5,
        pfe=jnp.ones(geo.rho_face_norm.shape) * 1.6,
        quasilinear_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.Rmaj,  # TODO
        gyrobohm_flux_reference_length=geo.Rmin,  # TODO
    )


if __name__ == '__main__':
  absltest.main()
