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
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

# pylint: disable=invalid-name


class PedestalModelOutputTest(parameterized.TestCase):

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

  def test_to_internal_boundary_conditions_shapes_and_values(self):
    idx = jnp.argmin(
        jnp.abs(self.geo.rho_norm - self.pedestal_model_output.rho_norm_ped_top)
    )
    ibc = self.pedestal_model_output.to_internal_boundary_conditions(
        self.geo,
        core_profiles=mock.create_autospec(state.CoreProfiles, instance=True),
    )
    self.assertEqual(ibc.T_i.shape, (self.geo.rho_norm.shape[0],))
    self.assertEqual(ibc.T_e.shape, (self.geo.rho_norm.shape[0],))
    self.assertEqual(ibc.n_e.shape, (self.geo.rho_norm.shape[0],))

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
    turbulent_core = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face),
    )
    turbulent_pedestal = transport_coeffs_lib.TransportCoeffs.zeros(self.geo)
    bgb_output = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face),
    )
    turbulent = transport_coeffs_lib.TurbulentTransport(
        core=turbulent_core,
        pedestal=turbulent_pedestal,
        core_components={'bohm_gyrobohm': bgb_output},
        pedestal_components={},
    )
    neoclassical = transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face) * 2.0,
        v_face_el_ware=jnp.ones(n_face),
    )
    pereverzev = transport_coeffs_lib.PereverzevTransport(
        chi_face_ion=jnp.ones(n_face) * 30.0,
        chi_face_el=jnp.ones(n_face) * 30.0,
        full_v_heat_face_ion=jnp.ones(n_face) * 15.0,
        full_v_heat_face_el=jnp.ones(n_face) * 12.0,
        d_face_el=jnp.ones(n_face) * 5.0,
        v_face_el=jnp.ones(n_face) * -3.0,
    )
    total = transport_coeffs_lib.sum_transport_coeffs(
        turbulent.total,
        neoclassical,
        pereverzev,
    )
    core_transport = state.CoreTransport(
        total=total,
        turbulent=turbulent,
        neoclassical=neoclassical,
        pereverzev=pereverzev,
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
        core_transport=core_transport,
        geo=self.geo,
        pedestal_runtime_params=pedestal_runtime_params,
    )
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )

    # Check turbulent transport is scaled correctly.
    np.testing.assert_allclose(
        modified_core_transport.turbulent.total.chi_face_el,
        jnp.where(pedestal_mask, 2.0, 1.0),
    )
    bgb_mod = modified_core_transport.turbulent.core_components['bohm_gyrobohm']
    self.assertIsInstance(bgb_mod, transport_coeffs_lib.TransportCoeffs)
    # Constituent core transport coefficients should NOT be scaled by the
    # pedestal model.
    np.testing.assert_allclose(
        bgb_mod.chi_face_el,
        bgb_output.chi_face_el,
    )

    np.testing.assert_allclose(
        modified_core_transport.turbulent.total.chi_face_ion,
        jnp.where(pedestal_mask, 3.0, 1.0),
    )
    np.testing.assert_allclose(
        bgb_mod.chi_face_ion,
        bgb_output.chi_face_ion,
    )

    np.testing.assert_allclose(
        modified_core_transport.turbulent.total.d_face_el,
        jnp.where(pedestal_mask, 4.0, 1.0),
    )
    np.testing.assert_allclose(
        modified_core_transport.turbulent.total.v_face_el,
        jnp.where(pedestal_mask, 5.0, 1.0),
    )

    # Check Pereverzev transport is left unscaled and unclipped across all 6
    # channels so that diffusion and counter-convection remain balanced.
    assert modified_core_transport.pereverzev is not None
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.chi_face_ion,
        pereverzev.chi_face_ion,
    )
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.chi_face_el,
        pereverzev.chi_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.d_face_el,
        pereverzev.d_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.v_face_el,
        pereverzev.v_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.full_v_heat_face_ion,
        pereverzev.full_v_heat_face_ion,
    )
    np.testing.assert_allclose(
        modified_core_transport.pereverzev.full_v_heat_face_el,
        pereverzev.full_v_heat_face_el,
    )

    # Check total transport sums scaled turbulent + unscaled neoclassical +
    # unscaled pereverzev.
    np.testing.assert_allclose(
        modified_core_transport.total.chi_face_ion,
        jnp.where(pedestal_mask, 3.0, 1.0) + 1.0 + 30.0,
    )
    np.testing.assert_allclose(
        modified_core_transport.total.chi_face_el,
        jnp.where(pedestal_mask, 2.0, 1.0) + 1.0 + 30.0,
    )
    np.testing.assert_allclose(
        modified_core_transport.total.d_face_el,
        jnp.where(pedestal_mask, 4.0, 1.0) + 1.0 + 5.0,
    )
    np.testing.assert_allclose(
        modified_core_transport.total.v_face_el,
        jnp.where(pedestal_mask, 5.0, 1.0) + 2.0 + (-3.0),
    )

    # Check neoclassical transport is not affected.
    np.testing.assert_allclose(
        modified_core_transport.neoclassical.chi_face_ion,
        core_transport.neoclassical.chi_face_ion,
    )
    np.testing.assert_allclose(
        modified_core_transport.neoclassical.chi_face_el,
        core_transport.neoclassical.chi_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.neoclassical.d_face_el,
        core_transport.neoclassical.d_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.neoclassical.v_face_el,
        core_transport.neoclassical.v_face_el,
    )
    np.testing.assert_allclose(
        modified_core_transport.neoclassical.v_face_el_ware,
        core_transport.neoclassical.v_face_el_ware,
    )

  def test_modify_core_transport_scales_only_core_and_preserves_pedestal(self):
    n_face = self.geo.rho_face_norm.shape[0]
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )
    turbulent_core = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face),
    )
    turbulent_pedestal = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.where(pedestal_mask, 0.25, 0.0),
        chi_face_el=jnp.where(pedestal_mask, 0.35, 0.0),
        d_face_el=jnp.where(pedestal_mask, 0.15, 0.0),
        v_face_el=jnp.where(pedestal_mask, -0.05, 0.0),
    )
    turbulent = transport_coeffs_lib.TurbulentTransport(
        core=turbulent_core,
        pedestal=turbulent_pedestal,
    )
    neoclassical = transport_coeffs_lib.NeoclassicalTransport.zeros(self.geo)
    core_transport = state.CoreTransport(
        total=turbulent.total + neoclassical,
        turbulent=turbulent,
        neoclassical=neoclassical,
        pereverzev=None,
    )
    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.chi_max = jnp.array(1.0)
    pedestal_runtime_params.D_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_min = jnp.array(-1.0)
    pedestal_runtime_params.pedestal_top_smoothing_width = jnp.array(0.0)

    modified = self.pedestal_model_output.modify_core_transport(
        core_transport=core_transport,
        geo=self.geo,
        pedestal_runtime_params=pedestal_runtime_params,
    )
    self.assertIsNone(modified.pereverzev)
    # Pedestal component should be untouched.
    np.testing.assert_allclose(
        modified.turbulent.pedestal.chi_face_ion,
        turbulent_pedestal.chi_face_ion,
    )
    np.testing.assert_allclose(
        modified.turbulent.pedestal.chi_face_el,
        turbulent_pedestal.chi_face_el,
    )
    # Core component should be scaled by multipliers (3.0 for chi_i, 2.0 for
    # chi_e).
    np.testing.assert_allclose(
        modified.turbulent.core.chi_face_ion,
        jnp.where(pedestal_mask, 3.0, 1.0),
    )
    # Total should be scaled core + unscaled pedestal.
    np.testing.assert_allclose(
        modified.turbulent.total.chi_face_ion,
        jnp.where(pedestal_mask, 3.25, 1.0),
    )
    np.testing.assert_allclose(
        modified.turbulent.total.chi_face_el,
        jnp.where(pedestal_mask, 2.35, 1.0),
    )

  def test_modify_core_transport_smooths_across_core_and_pedestal(self):
    n_face = self.geo.rho_face_norm.shape[0]
    ped_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.7,
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
    pedestal_mask = self.geo.rho_face_norm > ped_output.rho_norm_ped_top

    turbulent_core = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.linspace(1.0, 2.0, n_face),
        chi_face_el=jnp.linspace(1.5, 2.5, n_face),
        d_face_el=jnp.linspace(0.5, 1.5, n_face),
        v_face_el=jnp.linspace(-0.5, 0.5, n_face),
    )
    turbulent_pedestal = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.where(pedestal_mask, 0.25, 0.0),
        chi_face_el=jnp.where(pedestal_mask, 0.35, 0.0),
        d_face_el=jnp.where(pedestal_mask, 0.15, 0.0),
        v_face_el=jnp.where(pedestal_mask, -0.05, 0.0),
    )
    turbulent = transport_coeffs_lib.TurbulentTransport(
        core=turbulent_core,
        pedestal=turbulent_pedestal,
    )
    neoclassical = transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=jnp.ones(n_face) * 0.1,
        chi_face_el=jnp.ones(n_face) * 0.1,
        d_face_el=jnp.ones(n_face) * 0.1,
        v_face_el=jnp.ones(n_face) * 0.2,
        v_face_el_ware=jnp.ones(n_face) * 0.1,
    )
    pereverzev = transport_coeffs_lib.PereverzevTransport(
        chi_face_ion=jnp.ones(n_face) * 5.0,
        chi_face_el=jnp.ones(n_face) * 5.0,
        full_v_heat_face_ion=jnp.ones(n_face) * 2.5,
        full_v_heat_face_el=jnp.ones(n_face) * 2.0,
        d_face_el=jnp.ones(n_face) * 1.0,
        v_face_el=jnp.ones(n_face) * -0.5,
    )
    core_transport = state.CoreTransport(
        total=transport_coeffs_lib.sum_transport_coeffs(
            turbulent.total, neoclassical, pereverzev
        ),
        turbulent=turbulent,
        neoclassical=neoclassical,
        pereverzev=pereverzev,
    )
    smoothing_width = jnp.array(0.1)
    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.chi_max = jnp.array(10.0)
    pedestal_runtime_params.D_e_max = jnp.array(10.0)
    pedestal_runtime_params.V_e_max = jnp.array(10.0)
    pedestal_runtime_params.V_e_min = jnp.array(-10.0)
    pedestal_runtime_params.pedestal_top_smoothing_width = smoothing_width

    modified = ped_output.modify_core_transport(
        core_transport=core_transport,
        geo=self.geo,
        pedestal_runtime_params=pedestal_runtime_params,
    )

    # 1. modified.turbulent.pedestal is identical to original pedestal.
    np.testing.assert_allclose(
        modified.turbulent.pedestal.chi_face_ion,
        turbulent_pedestal.chi_face_ion,
    )
    np.testing.assert_allclose(
        modified.turbulent.pedestal.chi_face_el,
        turbulent_pedestal.chi_face_el,
    )
    np.testing.assert_allclose(
        modified.turbulent.pedestal.d_face_el,
        turbulent_pedestal.d_face_el,
    )
    np.testing.assert_allclose(
        modified.turbulent.pedestal.v_face_el,
        turbulent_pedestal.v_face_el,
    )

    # Compute expected scaled core (without smoothing).
    scaled_core_chi_i = jnp.where(
        pedestal_mask,
        turbulent_core.chi_face_ion * 3.0,
        turbulent_core.chi_face_ion,
    )
    scaled_core_chi_e = jnp.where(
        pedestal_mask,
        turbulent_core.chi_face_el * 2.0,
        turbulent_core.chi_face_el,
    )
    scaled_core_d_e = jnp.where(
        pedestal_mask,
        turbulent_core.d_face_el * 4.0,
        turbulent_core.d_face_el,
    )
    scaled_core_v_e = jnp.where(
        pedestal_mask,
        turbulent_core.v_face_el * 5.0,
        turbulent_core.v_face_el,
    )

    smoothing_matrix = math_utils.gaussian_smoothing_matrix(
        self.geo.rho_face_norm,
        smoothing_width,
        active=pedestal_model_output._pedestal_top_smoothing_mask(
            self.geo.rho_face_norm, ped_output.rho_norm_ped_top, smoothing_width
        ),
    )
    # Ensure smoothing matrix is non-trivial.
    self.assertFalse(np.allclose(smoothing_matrix, np.eye(n_face)))

    expected_total_chi_i = smoothing_matrix @ (
        scaled_core_chi_i + turbulent_pedestal.chi_face_ion
    )
    expected_total_chi_e = smoothing_matrix @ (
        scaled_core_chi_e + turbulent_pedestal.chi_face_el
    )
    expected_total_d_e = smoothing_matrix @ (
        scaled_core_d_e + turbulent_pedestal.d_face_el
    )
    expected_total_v_e = smoothing_matrix @ (
        scaled_core_v_e + turbulent_pedestal.v_face_el
    )

    # 2. modified.turbulent.total equals
    # smoothing_matrix @ (scaled_core + pedestal).
    np.testing.assert_allclose(
        modified.turbulent.total.chi_face_ion,
        expected_total_chi_i,
    )
    np.testing.assert_allclose(
        modified.turbulent.total.chi_face_el,
        expected_total_chi_e,
    )
    np.testing.assert_allclose(
        modified.turbulent.total.d_face_el,
        expected_total_d_e,
    )
    np.testing.assert_allclose(
        modified.turbulent.total.v_face_el,
        expected_total_v_e,
    )

    # Check modified.turbulent.core equals expected_total - pedestal.
    np.testing.assert_allclose(
        modified.turbulent.core.chi_face_ion,
        expected_total_chi_i - turbulent_pedestal.chi_face_ion,
    )

    # 3. modified.total equals
    # modified.turbulent.total + neoclassical + pereverzev.
    expected_sum = transport_coeffs_lib.sum_transport_coeffs(
        modified.turbulent.total, neoclassical, pereverzev
    )
    np.testing.assert_allclose(
        modified.total.chi_face_ion,
        expected_sum.chi_face_ion,
    )
    np.testing.assert_allclose(
        modified.total.chi_face_el,
        expected_sum.chi_face_el,
    )
    np.testing.assert_allclose(
        modified.total.d_face_el,
        expected_sum.d_face_el,
    )
    np.testing.assert_allclose(
        modified.total.v_face_el,
        expected_sum.v_face_el,
    )

  def test_to_internal_boundary_conditions_tanh_profiles(self):
    geo = circular_geometry.CircularConfig(n_rho=25).build_geometry()
    rho_ped_top = float(geo.rho_norm[22])
    core_profiles = _mtanh_core_profiles(geo, geo.rho_norm, 1.0)
    ped_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_ped_top,
        T_i_ped=5.0,
        T_e_ped=4.0,
        n_e_ped=0.7e20,
    )
    ibc_tanh = ped_output.to_internal_boundary_conditions(
        geo,
        core_profiles=core_profiles,
        pedestal_profile_form=(
            pedestal_runtime_params_lib.PedestalProfileForm.MTANH
        ),
    )
    ped_mask = pedestal_model_output.pedestal_cell_mask(
        geo.rho_face_norm, rho_ped_top
    )
    for name, top_v, sep_v in (
        ('T_i', 5.0, 0.1),
        ('T_e', 4.0, 0.08),
        ('n_e', 0.7e20, 1e19),
    ):
      with self.subTest(name):
        vals = getattr(ibc_tanh, name)
        np.testing.assert_array_equal(vals[~ped_mask], 0.0)
        ped_vals = vals[ped_mask]
        np.testing.assert_allclose(ped_vals[0], top_v, rtol=1e-10)
        self.assertTrue(bool(jnp.all(jnp.diff(ped_vals) < 0.0)))
        self.assertTrue(bool(jnp.all(ped_vals > sep_v)))

  @parameterized.parameters(0.925, 0.93)
  def test_interfacial_face_is_in_core_domain(self, rho_ped_top):
    geo = circular_geometry.CircularConfig(n_rho=100).build_geometry()
    cell_mask = pedestal_model_output.pedestal_cell_mask(
        geo.rho_face_norm, rho_ped_top
    )
    face_mask = pedestal_model_output.pedestal_face_mask(
        geo.rho_face_norm, rho_ped_top
    )
    first_ped_cell = int(np.argmax(np.asarray(cell_mask)))
    self.assertFalse(bool(face_mask[first_ped_cell]))

  @parameterized.parameters(1.25, 1.5, 2.0)
  def test_tanh_ibc_matches_across_grid_packings(self, packing):
    rho_ped_top = 0.93
    ped_eval = np.linspace(0.94, 0.99, 50)

    def _eval_ped_profile(p):
      geo = circular_geometry.CircularConfig(
          face_centers=1.0 - (1.0 - np.linspace(0.0, 1.0, 201)) ** p
      ).build_geometry()
      ibc = _mtanh_ibc(
          geo, rho_ped_top, _mtanh_core_profiles(geo, geo.rho_norm**2, 2.0)
      )
      mask = pedestal_model_output.pedestal_cell_mask(
          geo.rho_face_norm, rho_ped_top
      )
      return np.interp(ped_eval, geo.rho_norm[mask], ibc.T_i[mask])

    np.testing.assert_allclose(
        _eval_ped_profile(packing), _eval_ped_profile(1.0), atol=2e-2
    )

  def test_tanh_ibc_no_pedestal_grad_is_finite(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    no_ped = pedestal_model_output.PedestalModelOutput.no_pedestal()

    def _sum_ti(psi):
      ibc = no_ped.to_internal_boundary_conditions(
          geo,
          core_profiles=_mtanh_core_profiles(geo, psi, 2.0),
          pedestal_profile_form=(
              pedestal_runtime_params_lib.PedestalProfileForm.MTANH
          ),
      )
      return jnp.sum(ibc.T_i)

    grad_psi = jax.grad(_sum_ti)(geo.rho_norm**2)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_psi))))

  def test_tanh_ibc_first_pinned_cell(self):
    with self.subTest('equals_top_value_when_cell_center_at_ped_top'):
      # With psi linear in rho, psi interpolated at a cell center equals the
      # cell value, so the mtanh there is exactly T_i_ped.
      geo = circular_geometry.CircularConfig(n_rho=100).build_geometry()
      rho_ped_top = float(geo.rho_norm[93])
      ibc = _mtanh_ibc(
          geo, rho_ped_top, _mtanh_core_profiles(geo, geo.rho_norm, 1.0)
      )
      first = int(
          np.argmax(
              pedestal_model_output.pedestal_cell_mask(
                  geo.rho_face_norm, rho_ped_top
              )
          )
      )
      self.assertEqual(first, 93)
      np.testing.assert_allclose(ibc.T_i[first], 4.0, rtol=1e-10)
    with self.subTest('within_bound_at_half_cell_offset'):
      # With the pedestal top on a face, the first pinned cell center lies half
      # a cell inside it, the worst case.
      geo = circular_geometry.CircularConfig(n_rho=25).build_geometry()
      rho_ped_top = float(geo.rho_face_norm[24])
      ibc = _mtanh_ibc(
          geo, rho_ped_top, _mtanh_core_profiles(geo, geo.rho_norm**2, 2.0)
      )
      bound = (1.0 - np.tanh(2.0)) / (np.tanh(1.0) + np.tanh(2.0))
      self.assertBetween(float(ibc.T_i[23]), 4.0, 4.0 + bound * (4.0 - 0.1))


def _mtanh_core_profiles(
    geo: geometry.Geometry, psi: jax.Array | np.ndarray, psi_right_grad: float
) -> state.CoreProfiles:
  """Returns core profiles with the given psi and flat T_i, T_e, n_e."""
  core_profiles = mock.MagicMock(spec=state.CoreProfiles)
  core_profiles.psi = cell_variable.CellVariable(
      value=psi,
      face_centers=geo.rho_face_norm,
      right_face_constraint=None,
      right_face_grad_constraint=jnp.array(psi_right_grad),
  )
  for name, top_v, sep_v in (
      ('T_i', 4.0, 0.1),
      ('T_e', 4.0, 0.08),
      ('n_e', 1e20, 1e19),
  ):
    setattr(
        core_profiles,
        name,
        cell_variable.CellVariable(
            value=jnp.full_like(geo.rho_norm, top_v),
            face_centers=geo.rho_face_norm,
            right_face_constraint=jnp.array(sep_v),
            right_face_grad_constraint=None,
        ),
    )
  return core_profiles


def _mtanh_ibc(
    geo: geometry.Geometry,
    rho_norm_ped_top: float,
    core_profiles: state.CoreProfiles,
):
  """Returns the MTANH internal boundary conditions for T_i_ped = 4."""
  ped_out = pedestal_model_output.PedestalModelOutput(
      rho_norm_ped_top=rho_norm_ped_top, T_i_ped=4.0, T_e_ped=4.0, n_e_ped=1e20
  )
  return ped_out.to_internal_boundary_conditions(
      geo,
      core_profiles=core_profiles,
      pedestal_profile_form=pedestal_runtime_params_lib.PedestalProfileForm.MTANH,
  )


if __name__ == '__main__':
  absltest.main()
