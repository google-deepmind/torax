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
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import constants as constants_module
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import quasilinear_transport_model
from torax._src.transport_model import runtime_params as transport_params


constants = constants_module.CONSTANTS
jax.config.update('jax_enable_x64', True)


class QuasilinearTransportModelTest(parameterized.TestCase):

  # pylint: disable=invalid-name

  @parameterized.product(
      A_n_and_pfe=[
          # Zero density gradient (singular point for 1/A_n; purely convective).
          (0.0, 0.5),
          # Down-gradient within both |A_n| < An_min and pfe < smooth_width
          # transition zones (blended D and V in smooth mode).
          (0.02, 0.01),
          # Up-gradient transport (A_n * pfe < 0; purely convective).
          (0.5, -0.5),
          # Down-gradient above both thresholds (purely diffusive).
          (0.5, 0.5),
      ],
      DV_effective_smooth_width=[0.0, 0.02],
  )
  def test_dv_effective_conserves_flux_and_non_negative_d(
      self, A_n_and_pfe, DV_effective_smooth_width
  ):
    A_n_target, pfe_val = A_n_and_pfe
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    L_ref = 3.0
    An_min = 0.05
    n_e_0 = 1.0e20
    slope = -A_n_target * n_e_0 * geo.rho_b / L_ref
    n_e = cell_variable.CellVariable(
        value=n_e_0 + slope * geo.rho_norm,
        face_centers=geo.rho_face_norm,
        left_face_grad_constraint=jnp.asarray(slope),
        right_face_constraint=n_e_0 + slope,
        right_face_grad_constraint=None,
    )
    pfe = jnp.full_like(geo.rho_face_norm, pfe_val)
    pfe_SI = pfe * (n_e.face_value() / geo.a_minor) * 4.0

    d_face_el, v_face_el = quasilinear_transport_model.calculate_dv_effective(
        particle_flux_SI=pfe_SI,
        normalized_particle_flux=pfe,
        n_e=n_e,
        geo=geo,
        gradient_reference_length=L_ref,
        An_min=An_min,
        DV_effective_smooth_width=DV_effective_smooth_width,
        two_point_mask=two_point_mask,
    )
    reconstructed_flux = (
        -d_face_el
        * n_e.face_grad(two_point_mask=two_point_mask)
        * geo.g1_over_vpr2_face
        * geo.rho_b
        + v_face_el * n_e.face_value() * geo.g0_over_vpr_face * geo.rho_b
    )
    np.testing.assert_allclose(
        reconstructed_flux, pfe_SI, rtol=1e-12, atol=1e-8
    )
    self.assertTrue(np.all(d_face_el >= 0.0))
    if A_n_target * pfe_val <= 0.0:
      np.testing.assert_allclose(d_face_el, 0.0, atol=1e-15)

  def test_calculate_chiGB(self):
    """Tests that chiGB is calculated correctly."""

    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=np.array(1.0),
        reference_magnetic_field=1.0,
        reference_mass=1.0,
        reference_length=1.0,
    )
    chi_GB_expected = (
        (1.0 * constants.m_amu) ** 0.5
        / (constants.q_e) ** 2
        * (1.0 * constants.keV_to_J) ** 1.5
    )
    np.testing.assert_allclose(chiGB, chi_GB_expected)

  def test_calculate_normalized_logarithmic_gradient(self):
    """Tests that calculate_normalized_logarithmic_gradient is calculated correctly."""
    dummy_cell_variable = cell_variable.CellVariable(
        value=jnp.array([2.0, 1.0]),
        face_centers=jnp.array([0.0, 1.0, 2.0]),
        right_face_constraint=jnp.array(0.5),
        right_face_grad_constraint=None,
    )
    radial_coordinate = jnp.array([0.0, 1.0])
    radial_face_coordinate = jnp.array([-0.5, 0.5, 1.5])
    # pylint: disable=protected-access
    normalized_logarithmic_gradient = (
        quasilinear_transport_model.calculate_normalized_logarithmic_gradient(
            var=dummy_cell_variable,
            radial_coordinate=radial_coordinate,
            radial_face_coordinate=radial_face_coordinate,
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

  def test_calculate_normalized_logarithmic_gradient_preserves_negative_sign_near_zero(
      self,
  ):
    small_hollow_var = cell_variable.CellVariable(
        value=jnp.array([1.0, 1.0 + 0.25 * constants.eps]),
        face_centers=jnp.array([0.0, 1.0, 2.0]),
        right_face_constraint=jnp.array(1.0 + 0.5 * constants.eps),
        right_face_grad_constraint=None,
    )
    hollow_grad = (
        quasilinear_transport_model.calculate_normalized_logarithmic_gradient(
            var=small_hollow_var,
            radial_coordinate=jnp.array([0.0, 1.0]),
            radial_face_coordinate=jnp.array([-0.5, 0.5, 1.5]),
            reference_length=jnp.array(1.0),
        )
    )
    np.testing.assert_allclose(
        hollow_grad,
        np.array([1.0, -1.0, -1.0]) * constants.eps,
    )

  def test_apply_fast_ion_stabilization_multiplier_scales_correction(self):
    """Tests that the multiplier scales (factor-1), not the whole factor."""
    n_rho = 5
    smag = jnp.ones(n_rho)
    q = 1.5 * jnp.ones(n_rho)
    n_e = 1e20 * jnp.ones(n_rho)
    T_e = jnp.ones(n_rho)
    T_fi = 15.0 * T_e
    lref_over_lt_fi = 30.0 * jnp.ones(n_rho)
    lref_over_lti = 5.0 * jnp.ones(n_rho)
    rho_face = jnp.linspace(0.0, 1.0, n_rho)

    def _make_face_cv(face_value):
      cv = cell_variable.CellVariable(
          value=jnp.zeros(n_rho - 1),
          face_centers=rho_face,
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(0.0),
      )
      cv = mock.create_autospec(cv, instance=True)
      cv.face_value.return_value = face_value
      return cv

    key = 'icrh_He3'
    fast_ion_zero = mock.MagicMock()
    fast_ion_zero.species = 'He3'
    fast_ion_zero.source = 'icrh'
    fast_ion_zero.n = _make_face_cv(jnp.zeros(n_rho))
    fast_ion_zero.T = _make_face_cv(T_fi)

    fast_ion_nonzero = mock.MagicMock()
    fast_ion_nonzero.species = 'He3'
    fast_ion_nonzero.source = 'icrh'
    fast_ion_nonzero.n = _make_face_cv(0.01 * n_e)
    fast_ion_nonzero.T = _make_face_cv(T_fi)

    core_profiles_zero = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        fast_ions=(fast_ion_zero,),
    )
    core_profiles_zero.n_e = _make_face_cv(n_e)
    core_profiles_zero.T_e = _make_face_cv(T_e)

    core_profiles_nonzero = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        fast_ions=(fast_ion_nonzero,),
    )
    core_profiles_nonzero.n_e = _make_face_cv(n_e)
    core_profiles_nonzero.T_e = _make_face_cv(T_e)

    gradients_base = quasilinear_transport_model.NormalizedLogarithmicGradients(
        lref_over_lti=lref_over_lti,
        lref_over_lte=jnp.zeros(n_rho),
        lref_over_lne=jnp.zeros(n_rho),
        lref_over_lni0=jnp.zeros(n_rho),
        lref_over_lni1=jnp.zeros(n_rho),
        fast_ion_gradients={key: {'lref_over_lt': lref_over_lt_fi}},
    )

    factor_zero = (
        quasilinear_transport_model._compute_fast_ion_stabilization_factor(
            core_profiles=core_profiles_zero,
            smag=smag,
            q=q,
            normalized_logarithmic_gradients=gradients_base,
        )
    )

    factor_nonzero = (
        quasilinear_transport_model._compute_fast_ion_stabilization_factor(
            core_profiles=core_profiles_nonzero,
            smag=smag,
            q=q,
            normalized_logarithmic_gradients=gradients_base,
        )
    )

    # Regardless of multiplier, if no fast ions, should be 1.
    np.testing.assert_allclose(factor_zero, jnp.ones(n_rho))
    self.assertTrue(jnp.all(factor_nonzero > 1.0))

    multiplier = 3.0
    transport = mock.create_autospec(
        transport_params.ComponentRuntimeParams,
        instance=True,
        fast_ion_stabilization=True,
        fast_ion_stabilization_model=(),
        fast_ion_stabilization_multiplier=multiplier,
    )
    result = quasilinear_transport_model.apply_fast_ion_stabilization(
        core_profiles=core_profiles_nonzero,
        smag=smag,
        q=q,
        normalized_logarithmic_gradients=gradients_base,
        transport=transport,
    )
    expected_factor = (factor_nonzero - 1) * multiplier + 1
    np.testing.assert_allclose(result, lref_over_lti / expected_factor)

  @mock.patch.object(
      quasilinear_transport_model,
      '_load_fi_stabilization_model',
  )
  @mock.patch.object(
      quasilinear_transport_model,
      '_get_default_fi_stabilization_model',
  )
  def test_per_species_model_selection(self, mock_get_default, mock_load_model):
    """Tests that per-species model map routes correctly."""
    n_rho = 5
    smag = jnp.ones(n_rho)
    q = 1.5 * jnp.ones(n_rho)
    n_e = 1e20 * jnp.ones(n_rho)
    T_e = jnp.ones(n_rho)
    T_fi = 15.0 * T_e
    lref_over_lt_fi = 30.0 * jnp.ones(n_rho)

    rho_face = jnp.linspace(0.0, 1.0, n_rho)

    def _make_face_cv(face_value):
      cv = cell_variable.CellVariable(
          value=jnp.zeros(n_rho - 1),
          face_centers=rho_face,
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(0.0),
      )
      cv = mock.create_autospec(cv, instance=True)
      cv.face_value.return_value = face_value
      return cv

    fast_ion_he3 = mock.MagicMock()
    fast_ion_he3.species = 'He3'
    fast_ion_he3.source = 'icrh'
    fast_ion_he3.n = _make_face_cv(0.01 * n_e)
    fast_ion_he3.T = _make_face_cv(T_fi)

    fast_ion_h = mock.MagicMock()
    fast_ion_h.species = 'H'
    fast_ion_h.source = 'nbi'
    fast_ion_h.n = _make_face_cv(0.01 * n_e)
    fast_ion_h.T = _make_face_cv(T_fi)

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        fast_ions=(fast_ion_he3, fast_ion_h),
    )
    core_profiles.n_e = _make_face_cv(n_e)
    core_profiles.T_e = _make_face_cv(T_e)

    gradients = quasilinear_transport_model.NormalizedLogarithmicGradients(
        lref_over_lti=jnp.zeros(n_rho),
        lref_over_lte=jnp.zeros(n_rho),
        lref_over_lne=jnp.zeros(n_rho),
        lref_over_lni0=jnp.zeros(n_rho),
        lref_over_lni1=jnp.zeros(n_rho),
        fast_ion_gradients={
            'icrh_He3': {'lref_over_lt': lref_over_lt_fi},
            'nbi_H': {'lref_over_lt': lref_over_lt_fi},
        },
    )

    dummy_model = mock.MagicMock()
    dummy_model.predict.return_value = (jnp.ones((n_rho, 1)), None)
    mock_get_default.return_value = dummy_model
    mock_load_model.return_value = dummy_model

    model_map = {'He3': 'custom_he3_model'}
    quasilinear_transport_model._compute_fast_ion_stabilization_factor(
        core_profiles=core_profiles,
        smag=smag,
        q=q,
        normalized_logarithmic_gradients=gradients,
        model_map=model_map,
    )

    mock_load_model.assert_called_once_with('custom_he3_model')
    mock_get_default.assert_called_once_with('H')

  @mock.patch.object(
      quasilinear_transport_model,
      '_get_default_fi_stabilization_model',
  )
  def test_he4_maps_to_he3_default(self, mock_get_default):
    """Tests that He4 uses the same default model as He3."""
    n_rho = 5
    smag = jnp.ones(n_rho)
    q = 1.5 * jnp.ones(n_rho)
    n_e = 1e20 * jnp.ones(n_rho)
    T_e = jnp.ones(n_rho)
    T_fi = 15.0 * T_e
    lref_over_lt_fi = 30.0 * jnp.ones(n_rho)
    rho_face = jnp.linspace(0.0, 1.0, n_rho)

    def _make_face_cv(face_value):
      cv = cell_variable.CellVariable(
          value=jnp.zeros(n_rho - 1),
          face_centers=rho_face,
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(0.0),
      )
      cv = mock.create_autospec(cv, instance=True)
      cv.face_value.return_value = face_value
      return cv

    fast_ion_he4 = mock.MagicMock()
    fast_ion_he4.species = 'He4'
    fast_ion_he4.source = 'icrh'
    fast_ion_he4.n = _make_face_cv(0.01 * n_e)
    fast_ion_he4.T = _make_face_cv(T_fi)

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        fast_ions=(fast_ion_he4,),
    )
    core_profiles.n_e = _make_face_cv(n_e)
    core_profiles.T_e = _make_face_cv(T_e)

    gradients = quasilinear_transport_model.NormalizedLogarithmicGradients(
        lref_over_lti=jnp.zeros(n_rho),
        lref_over_lte=jnp.zeros(n_rho),
        lref_over_lne=jnp.zeros(n_rho),
        lref_over_lni0=jnp.zeros(n_rho),
        lref_over_lni1=jnp.zeros(n_rho),
        fast_ion_gradients={
            'icrh_He4': {'lref_over_lt': lref_over_lt_fi},
        },
    )

    dummy_model = mock.MagicMock()
    dummy_model.predict.return_value = (jnp.ones((n_rho, 1)), None)
    mock_get_default.return_value = dummy_model

    quasilinear_transport_model._compute_fast_ion_stabilization_factor(
        core_profiles=core_profiles,
        smag=smag,
        q=q,
        normalized_logarithmic_gradients=gradients,
    )

    mock_get_default.assert_called_once_with('He4')

  def test_unsupported_species_raises(self):
    """Tests that an unsupported species raises ValueError."""
    quasilinear_transport_model._get_default_fi_stabilization_model.cache_clear()
    with self.assertRaises(ValueError):
      quasilinear_transport_model._get_default_fi_stabilization_model('Ar')


if __name__ == '__main__':
  absltest.main()
