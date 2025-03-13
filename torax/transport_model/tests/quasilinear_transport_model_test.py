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
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import constants as constants_module
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.pedestal_model import set_tped_nped
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


constants = constants_module.CONSTANTS
jax.config.update('jax_enable_x64', True)


def _get_model_inputs(transport: quasilinear_transport_model.RuntimeParams):
  """Returns the model inputs for testing."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  geo = geometry_pydantic_model.CircularConfig().build_geometry()
  sources = sources_pydantic_model.Sources()
  source_models = source_models_lib.SourceModels(
      sources=sources.source_model_config
  )
  pedestal = pedestal_pydantic_model.Pedestal()
  dynamic_runtime_params_slice = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport,
          sources=sources,
          torax_mesh=geo.torax_mesh,
          pedestal=pedestal,
      )(
          t=runtime_params.numerics.t_initial,
      )
  )
  static_slice = build_runtime_params.build_static_runtime_params_slice(
      runtime_params=runtime_params,
      sources=sources,
      torax_mesh=geo.torax_mesh,
  )
  core_profiles = initialization.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_slice,
      geo=geo,
      source_models=source_models,
  )
  pedestal_model = set_tped_nped.SetTemperatureDensityPedestalModel()
  pedestal_model_outputs = pedestal_model(
      dynamic_runtime_params_slice, geo, core_profiles
  )
  return (
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      pedestal_model_outputs,
  )


class QuasilinearTransportModelTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.quasilinear_transport_model` module."""

  # pylint: disable=invalid-name

  def test_quasilinear_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    transport = quasilinear_transport_model.RuntimeParams()

    transport_model = FakeQuasilinearTransportModel()
    (
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        pedestal_model_outputs,
    ) = _get_model_inputs(transport)
    core_transport = transport_model(
        dynamic_runtime_params_slice, geo, core_profiles, pedestal_model_outputs
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
        (np.sum(np.abs(core_transport.v_face_el)) == 0.0),
        expected_zero_v_face_el,
    )
    self.assertEqual(
        (np.sum(np.abs(core_transport.d_face_el)) == 0.0),
        expected_zero_d_face_el,
    )

  def test_calculate_chiGB(self):
    """Tests that chiGB is calculated correctly."""
    core_profiles = _get_dummy_core_profiles(
        value=jnp.array([1.0]), right_face_constraint=jnp.array(1.0)
    )
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.temp_ion.face_value(),
        reference_magnetic_field=1.0,
        reference_mass=1.0,
        reference_length=1.0,
    )
    chi_GB_expected = (
        (1.0 * constants.mp) ** 0.5
        / (constants.qe) ** 2
        * (1.0 * constants.keV2J) ** 1.5
    )
    np.testing.assert_allclose(chiGB, chi_GB_expected)

  def test_calculate_alpha(self):
    """Tests that alpha is calculated correctly."""
    core_profiles = _get_dummy_core_profiles(
        value=jnp.array([1.0]), right_face_constraint=jnp.array(1.0)
    )
    normalized_logarithmic_gradients = (
        quasilinear_transport_model.NormalizedLogarithmicGradients(
            lref_over_lti=np.array([0.0, 1.0]),
            lref_over_lte=np.array([0.0, 2.0]),
            lref_over_lne=np.array([0.0, 3.0]),
            lref_over_lni0=np.array([0.0, 4.0]),
            lref_over_lni1=np.array([0.0, 5.0]),
        )
    )
    alpha = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        nref=1e20,
        q=np.array(1.0),
        reference_magnetic_field=1.0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )

    alpha_expected = np.array([0, 32 * constants.keV2J * 1e20 * constants.mu0])
    np.testing.assert_allclose(alpha, alpha_expected)

  def test_calculate_normalized_logarithmic_gradient(self):
    """Tests that calculate_normalized_logarithmic_gradient is calculated correctly."""
    dummy_cell_variable = cell_variable.CellVariable(
        value=jnp.array([2.0, 1.0]),
        right_face_constraint=jnp.array(0.5),
        right_face_grad_constraint=None,
        dr=jnp.array(1.0),
    )
    radial_coordinate = jnp.array([0.0, 1.0])
    # pylint: disable=protected-access
    normalized_logarithmic_gradient = (
        quasilinear_transport_model.calculate_normalized_logarithmic_gradient(
            var=dummy_cell_variable,
            radial_coordinate=radial_coordinate,
            reference_length=jnp.array(1.0),
        )
    )
    # pylint: enable=protected-access
    normalized_logarithmic_gradient_expected = np.array(
        [constants.eps, 2.0 / 3.0, 2.0]
    )
    np.testing.assert_allclose(
        normalized_logarithmic_gradient,
        normalized_logarithmic_gradient_expected,
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
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    quasilinear_inputs = quasilinear_transport_model.QuasilinearInputs(
        chiGB=np.array(4.0),
        Rmin=np.array(0.5),
        Rmaj=np.array(1.0),
        lref_over_lti=np.array(1.1),
        lref_over_lte=np.array(1.2),
        lref_over_lne=np.array(1.3),
        lref_over_lni0=np.array(1.4),
        lref_over_lni1=np.array(1.5),
    )
    transport = dynamic_runtime_params_slice.transport
    # Assert required for pytype.
    assert isinstance(
        transport,
        quasilinear_transport_model.DynamicRuntimeParams,
    )
    return self._make_core_transport(
        qi=np.ones(geo.rho_face_norm.shape) * 0.4,
        qe=np.ones(geo.rho_face_norm.shape) * 0.5,
        pfe=np.ones(geo.rho_face_norm.shape) * 1.6,
        quasilinear_inputs=quasilinear_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=3.0,
        gyrobohm_flux_reference_length=1.0,
    )


def _get_dummy_core_profiles(value, right_face_constraint):
  """Returns dummy core profiles for testing."""
  geo = geometry_pydantic_model.CircularConfig().build_geometry()
  currents = state.Currents.zeros(geo)
  dummy_cell_variable = cell_variable.CellVariable(
      value=value,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=None,
      dr=jnp.array(1.0),
  )
  return state.CoreProfiles(
      temp_ion=dummy_cell_variable,
      temp_el=dummy_cell_variable,
      ne=dummy_cell_variable,
      ni=dummy_cell_variable,
      nimp=dummy_cell_variable,
      currents=currents,
      Zi=1.0,
      Zi_face=1.0,
      Zimp=1.0,
      Zimp_face=1.0,
      Ai=1.0,
      Aimp=1.0,
      nref=1.0,
      q_face=1.0,
      s_face=1.0,
      psi=dummy_cell_variable,
      psidot=dummy_cell_variable,
      vloop_lcfs=1.0,
  )


if __name__ == '__main__':
  absltest.main()
