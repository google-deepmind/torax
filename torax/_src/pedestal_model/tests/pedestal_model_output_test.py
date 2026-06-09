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
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name


class PedestalModelOutputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = mock.create_autospec(geometry.Geometry, instance=True)
    self.geo.rho_face_norm = jnp.linspace(0, 1, 10)
    self.geo.rho = jnp.linspace(0, 1, 9)
    self.geo.rho_norm = jnp.linspace(0, 1, 9)
    self.pedestal_model_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.99,
        T_i_ped=1.0,
        T_e_ped=1.1,
        n_e_ped=1.2e19,
        transport_multipliers=pedestal_model_output.TransportMultipliers(
            chi_e_multiplier=jnp.array(2.0),
            chi_i_multiplier=jnp.array(3.0),
            D_e_multiplier=jnp.array(4.0),
            v_e_multiplier=jnp.array(5.0),
        ),
    )

  def test_to_internal_boundary_conditions(self):
    ibc = self.pedestal_model_output.to_internal_boundary_conditions(self.geo)
    idx = jnp.argmin(
        jnp.abs(self.geo.rho_norm - self.pedestal_model_output.rho_norm_ped_top)
    )
    with self.subTest('T_i'):
      np.testing.assert_allclose(
          ibc.T_i[idx], self.pedestal_model_output.T_i_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_i), self.pedestal_model_output.T_i_ped
      )
    with self.subTest('T_e'):
      np.testing.assert_allclose(
          ibc.T_e[idx], self.pedestal_model_output.T_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_e), self.pedestal_model_output.T_e_ped
      )
    with self.subTest('n_e'):
      np.testing.assert_allclose(
          ibc.n_e[idx], self.pedestal_model_output.n_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.n_e), self.pedestal_model_output.n_e_ped
      )

  def test_modify_core_transport_applies_multipliers(self):
    n_face = self.geo.rho_face_norm.shape[0]
    core_transport = state.CoreTransport(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face),
        chi_face_el_bohm=jnp.ones(n_face),
        chi_face_el_gyrobohm=jnp.ones(n_face),
        chi_face_ion_bohm=jnp.ones(n_face),
        chi_face_ion_gyrobohm=jnp.ones(n_face),
        chi_face_el_itg=jnp.ones(n_face),
        chi_face_el_tem=jnp.ones(n_face),
        chi_face_el_etg=jnp.ones(n_face),
        chi_face_ion_itg=jnp.ones(n_face),
        chi_face_ion_tem=jnp.ones(n_face),
        d_face_el_itg=jnp.ones(n_face),
        d_face_el_tem=jnp.ones(n_face),
        v_face_el_itg=jnp.ones(n_face),
        v_face_el_tem=jnp.ones(n_face),
        chi_neo_i=jnp.ones(n_face),
        chi_neo_e=jnp.ones(n_face),
        D_neo_e=jnp.ones(n_face),
        V_neo_e=jnp.ones(n_face),
        V_neo_ware_e=jnp.ones(n_face),
        chi_face_ion_pereverzev=jnp.ones(n_face),
        chi_face_el_pereverzev=jnp.ones(n_face),
        full_v_heat_face_ion_pereverzev=jnp.ones(n_face),
        full_v_heat_face_el_pereverzev=jnp.ones(n_face),
        d_face_el_pereverzev=jnp.ones(n_face),
        v_face_el_pereverzev=jnp.ones(n_face),
    )

    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.chi_max = jnp.array(1.0)
    pedestal_runtime_params.D_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_min = jnp.array(-1.0)
    pedestal_runtime_params.pedestal_top_smoothing_width = jnp.array(0.0)

    modified_core_transport = self.pedestal_model_output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )

    # Check turbulent and Pereverzev transport is scaled correctly.
    for field_name in [
        'chi_face_el',
        'chi_face_el_bohm',
        'chi_face_el_gyrobohm',
        'chi_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 2.0, 1.0),
      )
    for field_name in [
        'chi_face_ion',
        'chi_face_ion_bohm',
        'chi_face_ion_gyrobohm',
        'chi_face_ion_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 3.0, 1.0),
      )
    for field_name in [
        'd_face_el',
        'd_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 4.0, 1.0),
      )
    for field_name in [
        'v_face_el',
        'v_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 5.0, 1.0),
      )

    # Check neoclassical transport is not affected.
    np.testing.assert_allclose(
        modified_core_transport.chi_neo_i,
        core_transport.chi_neo_i,
    )
    np.testing.assert_allclose(
        modified_core_transport.chi_neo_e,
        core_transport.chi_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.D_neo_e,
        core_transport.D_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.V_neo_e,
        core_transport.V_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.V_neo_ware_e,
        core_transport.V_neo_ware_e,
    )

  def test_to_internal_boundary_conditions_tanh_profiles(self):
    """Tests mtanh-shaped IBC when pedestal_profile_form=MTANH."""
    geo = circular_geometry.CircularConfig(n_rho=25).build_geometry()
    n_cell = geo.torax_mesh.nx

    # Use psi ~ rho^2 so the mapping is analytically known.
    psi_cell = geo.rho_norm**2

    rho_ped_top = 0.9
    T_i_ped = 5.0
    T_e_ped = 4.0
    n_e_ped = 0.7e20

    # Separatrix values (rightmost face).
    T_i_sep = 0.1
    T_e_sep = 0.08
    n_e_sep = 0.1e20

    face_centers = geo.rho_face_norm

    def _make_standard_cell_var(cell_vals, right_face_val):
      return cell_variable.CellVariable(
          value=cell_vals,
          face_centers=face_centers,
          right_face_constraint=jnp.array(right_face_val),
          right_face_grad_constraint=None,
      )

    core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    core_profiles.psi = cell_variable.CellVariable(
        value=psi_cell,
        face_centers=face_centers,
        right_face_constraint=None,
        right_face_grad_constraint=jnp.array(2.0),
    )
    core_profiles.T_i = _make_standard_cell_var(
        jnp.linspace(10.0, T_i_sep, n_cell), T_i_sep
    )
    core_profiles.T_e = _make_standard_cell_var(
        jnp.linspace(8.0, T_e_sep, n_cell), T_e_sep
    )
    core_profiles.n_e = _make_standard_cell_var(
        jnp.linspace(1.0e20, n_e_sep, n_cell), n_e_sep
    )

    ped_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_ped_top,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )

    ibc_tanh = ped_output.to_internal_boundary_conditions(
        geo,
        core_profiles=core_profiles,
        pedestal_profile_form=pedestal_runtime_params_lib.PedestalProfileForm.MTANH,
    )
    ibc_point = ped_output.to_internal_boundary_conditions(geo)

    ped_mask = geo.rho_norm >= rho_ped_top

    with self.subTest('tanh_has_multiple_nonzero_cells'):
      # mtanh should have non-zero values at multiple cells.
      n_nonzero_tanh = jnp.sum(jnp.abs(ibc_tanh.T_i) > 0.0)
      n_nonzero_point = jnp.sum(jnp.abs(ibc_point.T_i) > 0.0)
      self.assertGreater(int(n_nonzero_tanh), int(n_nonzero_point))

    with self.subTest('core_region_is_zero'):
      # Core cells (rho < rho_ped_top) should be exactly zero.
      core_mask = ~ped_mask
      np.testing.assert_allclose(ibc_tanh.T_i[core_mask], 0.0)
      np.testing.assert_allclose(ibc_tanh.T_e[core_mask], 0.0)
      np.testing.assert_allclose(ibc_tanh.n_e[core_mask], 0.0)

    with self.subTest('pedestal_values_are_positive'):
      # All values in the pedestal region should be positive.
      self.assertTrue(jnp.all(ibc_tanh.T_i[ped_mask] > 0.0))
      self.assertTrue(jnp.all(ibc_tanh.T_e[ped_mask] > 0.0))
      self.assertTrue(jnp.all(ibc_tanh.n_e[ped_mask] > 0.0))

    with self.subTest('profile_decreases_from_top_to_separatrix'):
      # T_i in the pedestal region should be monotonically decreasing.
      ped_T_i = ibc_tanh.T_i[ped_mask]
      if len(ped_T_i) > 1:
        self.assertTrue(jnp.all(jnp.diff(ped_T_i) <= 0.0))

    with self.subTest('exact_profile_comparison'):
      # Independently compute expected mtanh profiles from the formula:
      #   f(psi) = f_sep + a0 * [tanh(1) - tanh(2*(psi - psi_mid)/delta)]
      # with psi_top = interp(rho_ped_top), delta = (1 - psi_top) / 1.5,
      # psi_mid = 1 - delta / 2.

      # Compute psi_norm at cell centers using face_value() to match impl.
      psi_fv = core_profiles.psi.face_value()
      psi_norm_cell = (psi_cell - psi_fv[0]) / (psi_fv[-1] - psi_fv[0])

      # Derive delta from nearest cell to rho_ped_top (matching impl).
      ped_top_idx = jnp.argmin(jnp.abs(geo.rho_norm - rho_ped_top))
      psi_top = psi_norm_cell[ped_top_idx]
      delta = (1.0 - psi_top) / 1.5
      psi_mid = 1.0 - delta / 2.0

      tanh1 = jnp.tanh(1.0)
      tanh2 = jnp.tanh(2.0)

      def _expected_mtanh(val_top, val_sep):
        a0 = (val_top - val_sep) / (tanh1 + tanh2)
        profile = val_sep + a0 * (
            tanh1 - jnp.tanh(2.0 * (psi_norm_cell - psi_mid) / delta)
        )
        return jnp.where(ped_mask, profile, 0.0)

      expected_T_i = _expected_mtanh(T_i_ped, T_i_sep)
      expected_T_e = _expected_mtanh(T_e_ped, T_e_sep)
      expected_n_e = _expected_mtanh(n_e_ped, n_e_sep)

      np.testing.assert_allclose(ibc_tanh.T_i, expected_T_i, rtol=1e-5)
      np.testing.assert_allclose(ibc_tanh.T_e, expected_T_e, rtol=1e-5)
      np.testing.assert_allclose(ibc_tanh.n_e, expected_n_e, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
