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

from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import base_model as ibc_base_model
from torax._src.internal_boundary_conditions import builder as ibc_builder
from torax._src.internal_boundary_conditions import internal_boundary_conditions as ibc_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import pereverzev as pereverzev_lib
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib
from torax._src.transport_model import transport_model as transport_model_lib


class TransportCoefficientsBuilderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.disable_jit())
    self.geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()
    self.n_face = len(self.geo.rho_face_norm)
    self.core_profiles = mock.create_autospec(state.CoreProfiles, instance=True)

    # Dummy turbulent transport output.
    self.turbulent_total = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.ones(self.n_face) * 2.0,
        chi_face_el=jnp.ones(self.n_face) * 3.0,
        d_face_el=jnp.ones(self.n_face) * 0.5,
        v_face_el=jnp.zeros(self.n_face),
    )
    self.turbulent_transport = transport_coeffs_lib.TurbulentTransport(
        core=self.turbulent_total,
        pedestal=transport_coeffs_lib.TransportCoeffs.zeros(self.geo),
        core_components={'mock': self.turbulent_total},
        pedestal_components={},
    )
    self.transport_model = mock.create_autospec(
        transport_model_lib.TransportModel, instance=True
    )
    self.transport_model.return_value = self.turbulent_transport

    # Dummy neoclassical transport output.
    self.neoclassical_transport = transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=jnp.ones(self.n_face) * 0.1,
        chi_face_el=jnp.ones(self.n_face) * 0.05,
        d_face_el=jnp.ones(self.n_face) * 0.02,
        v_face_el=jnp.zeros(self.n_face),
        v_face_el_ware=jnp.zeros(self.n_face),
    )

    # Mock IBC model and builder output.
    self.ibc_model = mock.create_autospec(
        ibc_base_model.InternalBoundaryConditionModel, instance=True
    )
    self.mock_ibc = mock.create_autospec(
        ibc_lib.InternalBoundaryConditions, instance=True
    )
    self.mock_two_point_mask = jnp.zeros(self.n_face, dtype=bool)
    self.mock_ibc.get_two_point_face_mask.return_value = (
        self.mock_two_point_mask
    )
    self.enterContext(
        mock.patch.object(
            ibc_builder,
            'build_internal_boundary_conditions',
            return_value=self.mock_ibc,
        )
    )

    # Mock runtime params.
    self.runtime_params = mock.Mock()
    self.runtime_params.pedestal.mode = None

    # Mock pedestal transition state.
    self.pedestal_output = pedestal_model_output_lib.PedestalModelOutput(
        rho_norm_ped_top=jnp.asarray(0.7),
        T_i_ped=jnp.asarray(4.0),
        T_e_ped=jnp.asarray(4.0),
        n_e_ped=jnp.asarray(0.6e20),
    )
    self.pedestal_state = mock.create_autospec(
        pedestal_transition_state_lib.PedestalTransitionState, instance=True
    )
    self.pedestal_state.pedestal_model_output = self.pedestal_output
    self.pedestal_state.is_ibc_active.return_value = False

    # Dummy Pereverzev transport output.
    self.pereverzev_transport = transport_coeffs_lib.PereverzevTransport(
        chi_face_ion=jnp.ones(self.n_face) * 10.0,
        chi_face_el=jnp.ones(self.n_face) * 10.0,
        full_v_heat_face_ion=jnp.ones(self.n_face) * 5.0,
        full_v_heat_face_el=jnp.ones(self.n_face) * 4.0,
        d_face_el=jnp.ones(self.n_face) * 2.0,
        v_face_el=jnp.ones(self.n_face) * -1.0,
    )

  def test_combines_turbulent_and_neoclassical_without_pereverzev(self):
    result = transport_coefficients_builder.calculate_all_transport_coeffs(
        transport_model=self.transport_model,
        neoclassical_transport=self.neoclassical_transport,
        internal_boundary_condition_model=self.ibc_model,
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=self.pedestal_state,
        use_pereverzev=False,
    )

    self.assertIsNone(result.pereverzev)
    self.assertEqual(result.turbulent, self.turbulent_transport)
    self.assertEqual(result.neoclassical, self.neoclassical_transport)
    expected_total = transport_coeffs_lib.sum_transport_coeffs(
        self.turbulent_total, self.neoclassical_transport
    )
    np.testing.assert_allclose(
        result.total.chi_face_ion, expected_total.chi_face_ion
    )
    np.testing.assert_allclose(
        result.total.chi_face_el, expected_total.chi_face_el
    )
    np.testing.assert_allclose(result.total.d_face_el, expected_total.d_face_el)
    np.testing.assert_allclose(result.total.v_face_el, expected_total.v_face_el)

  def test_includes_pereverzev_transport_when_enabled(self):
    with mock.patch.object(
        pereverzev_lib,
        'calculate_pereverzev_transport',
        return_value=self.pereverzev_transport,
    ) as mock_calc_pereverzev:
      result = transport_coefficients_builder.calculate_all_transport_coeffs(
          transport_model=self.transport_model,
          neoclassical_transport=self.neoclassical_transport,
          internal_boundary_condition_model=self.ibc_model,
          runtime_params=self.runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          pedestal_transition_state=self.pedestal_state,
          use_pereverzev=True,
      )

    mock_calc_pereverzev.assert_called_once_with(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        self.mock_two_point_mask,
    )
    self.assertIsNotNone(result.pereverzev)
    expected_total = transport_coeffs_lib.sum_transport_coeffs(
        self.turbulent_total,
        self.neoclassical_transport,
        self.pereverzev_transport,
    )
    np.testing.assert_allclose(
        result.total.chi_face_ion, expected_total.chi_face_ion
    )
    np.testing.assert_allclose(
        result.total.chi_face_el, expected_total.chi_face_el
    )
    np.testing.assert_allclose(result.total.d_face_el, expected_total.d_face_el)
    np.testing.assert_allclose(result.total.v_face_el, expected_total.v_face_el)

  def test_masks_pereverzev_in_pedestal_when_ibc_is_active(self):
    self.pedestal_state.is_ibc_active.return_value = True
    ped_mask = self.geo.rho_face_norm >= 0.7
    core_mask = ~ped_mask

    with mock.patch.object(
        pereverzev_lib,
        'calculate_pereverzev_transport',
        return_value=self.pereverzev_transport,
    ):
      result = transport_coefficients_builder.calculate_all_transport_coeffs(
          transport_model=self.transport_model,
          neoclassical_transport=self.neoclassical_transport,
          internal_boundary_condition_model=self.ibc_model,
          runtime_params=self.runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          pedestal_transition_state=self.pedestal_state,
          use_pereverzev=True,
      )

    # Pedestal faces must be zeroed out across all channels.
    np.testing.assert_allclose(result.pereverzev.chi_face_ion[ped_mask], 0.0)
    np.testing.assert_allclose(result.pereverzev.chi_face_el[ped_mask], 0.0)
    np.testing.assert_allclose(result.pereverzev.d_face_el[ped_mask], 0.0)
    np.testing.assert_allclose(result.pereverzev.v_face_el[ped_mask], 0.0)
    np.testing.assert_allclose(
        result.pereverzev.full_v_heat_face_ion[ped_mask], 0.0
    )
    np.testing.assert_allclose(
        result.pereverzev.full_v_heat_face_el[ped_mask], 0.0
    )

    # Core faces must remain unmasked.
    np.testing.assert_allclose(
        result.pereverzev.chi_face_ion[core_mask],
        self.pereverzev_transport.chi_face_ion[core_mask],
    )

  def test_does_not_mask_pereverzev_when_ibc_is_inactive(self):
    self.pedestal_state.is_ibc_active.return_value = False
    ped_mask = self.geo.rho_face_norm >= 0.7

    with mock.patch.object(
        pereverzev_lib,
        'calculate_pereverzev_transport',
        return_value=self.pereverzev_transport,
    ):
      result = transport_coefficients_builder.calculate_all_transport_coeffs(
          transport_model=self.transport_model,
          neoclassical_transport=self.neoclassical_transport,
          internal_boundary_condition_model=self.ibc_model,
          runtime_params=self.runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          pedestal_transition_state=self.pedestal_state,
          use_pereverzev=True,
      )

    np.testing.assert_allclose(
        result.pereverzev.chi_face_ion[ped_mask],
        self.pereverzev_transport.chi_face_ion[ped_mask],
    )

  def test_applies_adaptive_transport_modification_in_adaptive_mode(self):
    self.runtime_params.pedestal.mode = (
        pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    )
    mock_modified_transport = mock.create_autospec(
        state.CoreTransport, instance=True
    )
    mock_ped_output = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput, instance=True
    )
    mock_ped_output.modify_core_transport.return_value = mock_modified_transport
    self.pedestal_state.pedestal_model_output = mock_ped_output

    result = transport_coefficients_builder.calculate_all_transport_coeffs(
        transport_model=self.transport_model,
        neoclassical_transport=self.neoclassical_transport,
        internal_boundary_condition_model=self.ibc_model,
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=self.pedestal_state,
        use_pereverzev=False,
    )

    self.assertEqual(result, mock_modified_transport)
    mock_ped_output.modify_core_transport.assert_called_once()

  def test_does_not_modify_core_transport_when_not_adaptive_mode(self):
    self.runtime_params.pedestal.mode = (
        pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
    )
    mock_ped_output = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput, instance=True
    )
    self.pedestal_state.pedestal_model_output = mock_ped_output

    transport_coefficients_builder.calculate_all_transport_coeffs(
        transport_model=self.transport_model,
        neoclassical_transport=self.neoclassical_transport,
        internal_boundary_condition_model=self.ibc_model,
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=self.pedestal_state,
        use_pereverzev=False,
    )

    mock_ped_output.modify_core_transport.assert_not_called()


if __name__ == '__main__':
  absltest.main()
