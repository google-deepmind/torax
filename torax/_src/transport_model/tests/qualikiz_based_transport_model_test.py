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
from collections.abc import Mapping
import dataclasses
from typing import Annotated, Any, Literal
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import register_model
from torax._src.transport_model import (
    runtime_params as transport_runtime_params_lib,
)
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib


def setUpModule():
  register_model.register_transport_model(QualikizBasedTransportModelConfig)


def _get_config_and_model_inputs(
    transport: Mapping[str, Any],
):
  """Returns the model inputs for testing."""
  config_dict = default_configs.get_default_config_dict()
  config_dict['transport'] = transport
  torax_config = config.ToraxConfig.from_dict(config_dict)
  core_profiles, runtime_params, geo = core_profile_helpers.make_initial_inputs(
      torax_config
  )
  two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
  transport_model = torax_config.transport.build_transport_model()
  return torax_config, (
      runtime_params,
      geo,
      core_profiles,
      pedestal_model_output_lib.IBCPedestalModelOutput.empty(),
      two_point_mask,
      transport_model.initial_state(runtime_params),
  )


class QualikizTransportModelTest(parameterized.TestCase):

  def test_qualikiz_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        'core_transport_models': {
            'qualikiz_based': {
                'model_name': 'qualikiz_based',
                'collisionality_multiplier': 1.0,
                'avoid_big_negative_s': True,
                'q_sawtooth_proxy': True,
            },
        },
    })
    transport_model = torax_config.transport.build_transport_model()

    core_transport, _ = transport_model(*model_inputs)
    expected_shape = model_inputs[1].rho_face_norm.shape
    self.assertEqual(core_transport.total.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.total.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.total.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.total.v_face_el.shape, expected_shape)

  def test_qualikiz_based_transport_model_prepare_qualikiz_inputs_shapes(self):
    """Tests that the Qualikiz-based inputs have the expected shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        'core_transport_models': {
            'qualikiz_based': {
                'model_name': 'qualikiz_based',
                'collisionality_multiplier': 1.0,
                'avoid_big_negative_s': True,
                'q_sawtooth_proxy': True,
                'smag_alpha_correction': True,
            },
        },
    })
    transport_model = (
        torax_config.transport.build_transport_model().core_transport_models[
            'qualikiz_based'
        ]
    )
    assert isinstance(
        transport_model,
        qualikiz_based_transport_model.QualikizBasedTransportModel,
    )
    runtime_params, geo, core_profiles, _, two_point_mask, _ = model_inputs
    qualikiz_params = runtime_params.transport.core_transport_model_params[
        'qualikiz_based'
    ]
    assert isinstance(
        qualikiz_params, qualikiz_based_transport_model.RuntimeParams
    )
    qualikiz_inputs = transport_model._prepare_qualikiz_inputs(
        transport=qualikiz_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    # 1D array qualikiz_inputs
    vector_keys = [
        'Z_eff_face',
        'Ati',
        'Ate',
        'Ane',
        'Ani0',
        'Ani1',
        'q',
        'smag',
        'x',
        'Ti_Te',
        'log_nu_star_face',
        'normni',
        'chiGB',
    ]
    expected_vector_length = geo.rho_face_norm.shape[0]
    for key in vector_keys:
      self.assertEqual(
          getattr(qualikiz_inputs, key).shape, (expected_vector_length,)
      )

  def test_gamma_e_is_bounded_near_axis(self):
    torax_config, model_inputs = _get_config_and_model_inputs({
        'core_transport_models': {
            'qualikiz_based': {
                'model_name': 'qualikiz_based',
                'rotation_mode': 'full_radius',
            },
        },
    })
    transport_model = (
        torax_config.transport.build_transport_model().core_transport_models[
            'qualikiz_based'
        ]
    )
    assert isinstance(
        transport_model,
        qualikiz_based_transport_model.QualikizBasedTransportModel,
    )
    runtime_params, geo, core_profiles, _, two_point_mask, _ = model_inputs
    qualikiz_params = runtime_params.transport.core_transport_model_params[
        'qualikiz_based'
    ]
    assert isinstance(
        qualikiz_params, qualikiz_based_transport_model.RuntimeParams
    )
    qualikiz_inputs = transport_model._prepare_qualikiz_inputs(
        transport=qualikiz_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    # Normalized ExB shear is O(0.1) for these profiles. Dividing by r on the
    # magnetic axis previously produced values of order 1e5 near the axis.
    self.assertLess(np.max(np.abs(qualikiz_inputs.gamma_E_QLK)), 1.0)

  @parameterized.parameters((0.001,), (0.1,))
  def test_collisionality_max(self, max_normalized_collisionality):
    """Checks that the max collisionality is capped."""
    log_cap = jnp.log10(max_normalized_collisionality)
    # Get uncapped inputs (max_normalized_collisionality=inf).
    torax_config, uncapped_inputs = _get_config_and_model_inputs({
        'core_transport_models': {
            'qualikiz_based': {
                'model_name': 'qualikiz_based',
            },
        },
    })
    # Get capped inputs.
    _, capped_inputs = _get_config_and_model_inputs({
        'core_transport_models': {
            'qualikiz_based': {
                'model_name': 'qualikiz_based',
                'max_normalized_collisionality': max_normalized_collisionality,
            },
        },
    })
    transport_model = (
        torax_config.transport.build_transport_model().core_transport_models[
            'qualikiz_based'
        ]
    )
    assert isinstance(
        transport_model,
        qualikiz_based_transport_model.QualikizBasedTransportModel,
    )
    runtime_params_uncapped, geo, core_profiles, _, two_point_mask, _ = (
        uncapped_inputs
    )
    runtime_params_capped = capped_inputs[0]

    qualikiz_params_uncapped = (
        runtime_params_uncapped.transport.core_transport_model_params[
            'qualikiz_based'
        ]
    )
    qualikiz_params_capped = (
        runtime_params_capped.transport.core_transport_model_params[
            'qualikiz_based'
        ]
    )
    assert isinstance(
        qualikiz_params_uncapped, qualikiz_based_transport_model.RuntimeParams
    )
    assert isinstance(
        qualikiz_params_capped, qualikiz_based_transport_model.RuntimeParams
    )

    uncapped = transport_model._prepare_qualikiz_inputs(
        transport=qualikiz_params_uncapped,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    capped = transport_model._prepare_qualikiz_inputs(
        transport=qualikiz_params_capped,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    # Precondition: some uncapped values must exceed the cap.
    above_cap = uncapped.log_nu_star_face > log_cap
    self.assertTrue(jnp.any(above_cap))
    # Where uncapped exceeds cap, capped should equal the cap.
    np.testing.assert_allclose(
        capped.log_nu_star_face[above_cap],
        jnp.full_like(capped.log_nu_star_face[above_cap], log_cap),
    )
    # Where uncapped is at or below cap, capped should be unchanged.
    np.testing.assert_allclose(
        capped.log_nu_star_face[~above_cap],
        uncapped.log_nu_star_face[~above_cap],
    )

  def test_qualikiz_output_to_output_dict(self):
    n_face = 10
    ql_coeffs = qualikiz_based_transport_model.QualikizTransportModelOutput(
        chi_face_ion=jnp.ones((1, n_face)) * 1.0,
        chi_face_el=jnp.ones((1, n_face)) * 2.0,
        d_face_el=jnp.ones((1, n_face)) * 0.5,
        v_face_el=jnp.ones((1, n_face)) * -0.1,
        chi_face_ion_itg=jnp.ones((1, n_face)) * 0.8,
        chi_face_ion_tem=jnp.ones((1, n_face)) * 0.2,
        chi_face_el_itg=jnp.ones((1, n_face)) * 1.0,
        chi_face_el_tem=jnp.ones((1, n_face)) * 0.5,
        chi_face_el_etg=jnp.ones((1, n_face)) * 0.5,
        d_face_el_itg=jnp.ones((1, n_face)) * 0.3,
        d_face_el_tem=jnp.ones((1, n_face)) * 0.2,
        v_face_el_itg=jnp.ones((1, n_face)) * -0.05,
        v_face_el_tem=jnp.ones((1, n_face)) * -0.05,
    )
    context = output_grid_context.OutputGridContext(
        times=np.array([0.0]),
        rho_face_norm=np.linspace(0, 1, n_face),
        rho_cell_norm=np.linspace(0, 1, n_face - 1),
        rho_cell_plus_boundaries_norm=np.linspace(0, 1, n_face + 1),
    )
    out = ql_coeffs.to_output_dict(context)
    self.assertIn(output_keys.CHI_TURB_I, out)
    self.assertIn(output_keys.CHI_ITG_I, out)
    self.assertIn(output_keys.CHI_TEM_I, out)
    self.assertIn(output_keys.CHI_ITG_E, out)
    self.assertIn(output_keys.CHI_TEM_E, out)
    self.assertIn(output_keys.CHI_ETG_E, out)
    self.assertIn(output_keys.D_ITG_E, out)
    self.assertIn(output_keys.D_TEM_E, out)
    self.assertIn(output_keys.V_ITG_E, out)
    self.assertIn(output_keys.V_TEM_E, out)


@dataclasses.dataclass(frozen=True, eq=False)
class FakeQualikizBasedTransportModel(
    qualikiz_based_transport_model.QualikizBasedTransportModel
):
  """Fake QualikizBasedTransportModel for testing purposes."""

  def call_implementation(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      component_state: transport_state_lib.ComponentTransportState,
  ) -> tuple[
      transport_coeffs.TransportCoeffs,
      transport_state_lib.ComponentTransportState,
  ]:
    # Assert required for pytype.
    assert isinstance(
        transport_runtime_params,
        qualikiz_based_transport_model.RuntimeParams,
    )

    qualikiz_inputs = self._prepare_qualikiz_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    return (
        self._make_transport_coeffs(
            qi=jnp.ones(geo.rho_face_norm.shape) * 0.4,
            qe=jnp.ones(geo.rho_face_norm.shape) * 0.5,
            pfe=jnp.ones(geo.rho_face_norm.shape) * 1.6,
            qualikiz_inputs=qualikiz_inputs,
            transport=transport_runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            two_point_mask=two_point_mask,
        ),
        component_state,
    )


# pylint: disable=invalid-name
class QualikizBasedTransportModelConfig(
    qualikiz_based_transport_model.QualikizBasedTransportModelConfig
):
  """Model for testing the Qualikiz-based transport model."""

  model_name: Annotated[
      Literal['qualikiz_based'], torax_pydantic.JAX_STATIC
  ] = 'qualikiz_based'
  rotation_mode: Annotated[
      qualikiz_based_transport_model.RotationMode, torax_pydantic.JAX_STATIC
  ] = qualikiz_based_transport_model.RotationMode.HALF_RADIUS

  def build_transport_model(
      self,
  ) -> FakeQualikizBasedTransportModel:
    return FakeQualikizBasedTransportModel()


if __name__ == '__main__':
  absltest.main()
