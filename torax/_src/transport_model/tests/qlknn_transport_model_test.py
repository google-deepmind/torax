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
import jax.numpy as jnp
import numpy.testing as npt
from torax._src.geometry import geometry
from torax._src.transport_model import qlknn_10d
from torax._src.transport_model import qlknn_model_wrapper
from torax._src.transport_model import qlknn_transport_model
from torax._src.transport_model import qualikiz_based_transport_model


def _make_mock_qualikiz_inputs(n_rho: int = 10):
  """Create mock QualikizInputs for testing."""
  return qualikiz_based_transport_model.QualikizInputs(
      Z_eff_face=jnp.ones(n_rho),
      lref_over_lti=jnp.ones(n_rho),
      lref_over_lte=jnp.ones(n_rho),
      lref_over_lne=jnp.ones(n_rho),
      lref_over_lni0=jnp.ones(n_rho),
      lref_over_lni1=jnp.ones(n_rho),
      q=jnp.linspace(1.0, 4.0, n_rho),
      smag=jnp.linspace(0.1, 2.0, n_rho),
      x=jnp.linspace(0.0, 1.0, n_rho),
      Ti_Te=jnp.ones(n_rho),
      log_nu_star_face=jnp.zeros(n_rho),
      normni=jnp.ones(n_rho),
      chiGB=jnp.ones(n_rho),
      Rmaj=jnp.array(6.2),
      Rmin=jnp.array(2.0),
      alpha=jnp.zeros(n_rho),
      epsilon=jnp.linspace(0.0, 0.3, n_rho),
      gamma_E_GB=jnp.ones(n_rho) * 0.5,
      gamma_E_GB_poloidal_and_pressure=jnp.ones(n_rho) * 0.25,
      gamma_E_GB_toroidal=jnp.ones(n_rho) * 0.25,
      gamma_E_QLK=jnp.ones(n_rho) * 0.5,
      mach_toroidal=jnp.zeros(n_rho),
  )


def _make_mock_qualikiz_inputs_poloidal_only(n_rho: int = 10):
  """Create mock QualikizInputs with only poloidal/pressure contribution."""
  return qualikiz_based_transport_model.QualikizInputs(
      Z_eff_face=jnp.ones(n_rho),
      lref_over_lti=jnp.ones(n_rho),
      lref_over_lte=jnp.ones(n_rho),
      lref_over_lne=jnp.ones(n_rho),
      lref_over_lni0=jnp.ones(n_rho),
      lref_over_lni1=jnp.ones(n_rho),
      q=jnp.linspace(1.0, 4.0, n_rho),
      smag=jnp.linspace(0.1, 2.0, n_rho),
      x=jnp.linspace(0.0, 1.0, n_rho),
      Ti_Te=jnp.ones(n_rho),
      log_nu_star_face=jnp.zeros(n_rho),
      normni=jnp.ones(n_rho),
      chiGB=jnp.ones(n_rho),
      Rmaj=jnp.array(6.2),
      Rmin=jnp.array(2.0),
      alpha=jnp.zeros(n_rho),
      epsilon=jnp.linspace(0.0, 0.3, n_rho),
      gamma_E_GB=jnp.ones(n_rho) * 0.5,
      gamma_E_GB_poloidal_and_pressure=jnp.ones(n_rho) * 0.5,
      gamma_E_GB_toroidal=jnp.zeros(n_rho),
      gamma_E_QLK=jnp.ones(n_rho) * 0.5,
      mach_toroidal=jnp.zeros(n_rho),
  )


def _make_mock_model_output(n_rho: int = 10):
  """Create mock model output for testing."""
  return {
      'qi_itg': jnp.ones((n_rho, 1)),
      'qe_itg': jnp.ones((n_rho, 1)),
      'pfe_itg': jnp.ones((n_rho, 1)),
      'qi_tem': jnp.ones((n_rho, 1)),
      'qe_tem': jnp.ones((n_rho, 1)),
      'pfe_tem': jnp.ones((n_rho, 1)),
      'qe_etg': jnp.ones((n_rho, 1)),
      'gamma_max': jnp.ones((n_rho, 1)) * 0.5,
  }


def _make_mock_geometry(n_rho: int = 10):
  """Create a simple mock geometry for testing."""
  return mock.MagicMock(
      spec=geometry.Geometry,
      rho_face_norm=jnp.linspace(0.0, 1.0, n_rho),
  )


class QlknnTransportModelTest(parameterized.TestCase):

  def test_hash_and_eq(self):
    # Test that hash and eq are invariant to copying, so that they will work
    # correctly with jax's persistent cache
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    self.assertEqual(qlknn_1, qlknn_2)
    self.assertEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different(self):
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('baz', 'bar')
    self.assertNotEqual(qlknn_1, qlknn_2)
    self.assertNotEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertNotIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different_by_name(self):
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('foo', 'baz')
    self.assertNotEqual(qlknn_1, qlknn_2)
    self.assertNotEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertNotIn(qlknn_2, mock_persistent_jax_cache)

  @parameterized.named_parameters(
      ('itg', {'itg': False}),
      ('tem', {'tem': False}),
      ('etg', {'etg': False}),
      ('etg_and_itg', {'etg': False, 'itg': False}),
  )
  def test_filter_model_output(self, include_dict):
    """Tests that the model output is properly filtered."""

    shape = (26,)
    itg_keys = ['qi_itg', 'qe_itg', 'pfe_itg']
    tem_keys = ['qe_tem', 'qi_tem', 'pfe_tem']
    etg_keys = ['qe_etg']
    model_output = dict(
        [(k, jnp.ones(shape)) for k in itg_keys + tem_keys + etg_keys]
    )
    filtered_model_output = qlknn_transport_model._filter_model_output(
        model_output=model_output,
        include_ITG=include_dict.get('itg', True),
        include_TEM=include_dict.get('tem', True),
        include_ETG=include_dict.get('etg', True),
    )
    for key in itg_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('itg', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)
    for key in tem_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('tem', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)
    for key in etg_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('etg', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)

  def test_clip_inputs(self):
    """Tests that the inputs are properly clipped."""
    feature_scan = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [1.0, 2.8, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ])
    inputs_and_ranges = {
        'a': {'min': 0.0, 'max': 10.0},
        'b': {'min': 2.5, 'max': 10.0},
        'c': {'min': 0.0, 'max': 2.5},
        'd': {'min': 12.0, 'max': 15.0},
        'e': {'min': 0.0, 'max': 3.0},
    }
    clip_margin = 0.95
    expected = jnp.array([
        [1.0, 2.625, 2.375, 12.6, 2.85, 6.0, 7.0, 8.0, 9.0],
        [1.0, 2.8, 2.0, 12.6, 2.85, 6.0, 7.0, 8.0, 9.0],
    ])
    clipped_feature_scan = qlknn_transport_model.clip_inputs(
        feature_scan=feature_scan,
        inputs_and_ranges=inputs_and_ranges,
        clip_margin=clip_margin,
    )
    npt.assert_allclose(clipped_feature_scan, expected)

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_with_path(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='/my/foo.qlknn', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('/my/foo.qlknn', 'bar')

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_with_name_only(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('', 'bar')

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_without_path_or_name(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='', name='')
    mock_qlknn_model_wrapper.assert_called_once_with('', '')

  @mock.patch.object(qlknn_10d, 'QLKNN10D', autospec=True)
  def test_get_model_from_path_qlknn10d(self, mock_qlknn_qlknn10d):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='/foo/qlknn_hyper', name='bar')
    mock_qlknn_qlknn10d.assert_called_once_with('/foo/qlknn_hyper', 'bar')

  @mock.patch.object(qlknn_10d, 'QLKNN10D', autospec=True)
  def test_get_model_from_name_qlknn10d_fails(self, mock_qlknn_qlknn10d):
    """Tests that the model is loaded from the path."""
    with self.assertRaises(ValueError):
      qlknn_transport_model.get_model(path='', name='qlknn10D')
    mock_qlknn_qlknn10d.assert_not_called()

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_caching(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded only once."""
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('', 'bar')


class ShearSuppressionModelTest(parameterized.TestCase):
  """Tests for the shear suppression models in QLKNN transport."""

  def test_rotation_mode_off_returns_unchanged_output(self):
    """Tests that rotation mode OFF returns unchanged model output."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs(n_rho)
    geo = _make_mock_geometry(n_rho)

    result = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.OFF,
        shear_suppression_alpha=1.0,
        geo=geo,
    )

    for key in model_output:
      npt.assert_array_equal(result[key], model_output[key])

  @parameterized.named_parameters(
      ('alpha_0.5', 0.5),
      ('alpha_1.0', 1.0),
      ('alpha_2.0', 2.0),
  )
  def test_waltz_rule_suppresses_itg_and_tem_fluxes(self, alpha):
    """Tests that Waltz rule suppresses ITG and TEM fluxes."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs_poloidal_only(n_rho)
    geo = _make_mock_geometry(n_rho)

    result = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        shear_suppression_alpha=alpha,
        geo=geo,
    )

    for key in ['qi_itg', 'qe_itg', 'pfe_itg', 'qi_tem', 'qe_tem', 'pfe_tem']:
      self.assertTrue(
          bool(jnp.all(result[key][1:] <= model_output[key][1:])),
          f'Waltz rule should suppress {key} flux',
      )
    npt.assert_array_equal(result['qe_etg'][1:], model_output['qe_etg'][1:])

  def test_waltz_rule_larger_alpha_gives_more_suppression(self):
    """Tests that larger alpha gives more suppression in Waltz rule."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs_poloidal_only(n_rho)
    geo = _make_mock_geometry(n_rho)

    result_small_alpha = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        shear_suppression_alpha=0.5,
        geo=geo,
    )

    result_large_alpha = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        shear_suppression_alpha=2.0,
        geo=geo,
    )

    for key in ['qi_itg', 'qe_itg', 'pfe_itg', 'qi_tem', 'qe_tem', 'pfe_tem']:
      self.assertTrue(
          bool(
              jnp.all(
                  result_large_alpha[key][1:] <= result_small_alpha[key][1:]
              )
          ),
          f'Larger alpha should give more suppression for {key}',
      )

  def test_vandeplassche_rule_applies_rotation(self):
    """Tests that Van de Plassche 2020 rule applies rotation correction."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs(n_rho)
    geo = _make_mock_geometry(n_rho)

    result = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        shear_suppression_alpha=1.0,
        geo=geo,
    )

    for key in ['qi_itg', 'qe_itg', 'pfe_itg', 'qi_tem', 'qe_tem', 'pfe_tem']:
      self.assertFalse(
          bool(jnp.array_equal(result[key][1:], model_output[key][1:])),
          f'Van de Plassche rule should modify {key} flux',
      )
    npt.assert_array_equal(result['qe_etg'][1:], model_output['qe_etg'][1:])

  def test_half_radius_mode_only_affects_outer_region(self):
    """Tests that half_radius mode only applies rotation to outer region."""
    n_rho = 20
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs(n_rho)
    geo = _make_mock_geometry(n_rho)

    result = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.HALF_RADIUS,
        shear_suppression_alpha=1.0,
        geo=geo,
    )

    rho = geo.rho_face_norm
    inner_idx = (rho > 0.05) & (rho < 0.4)
    outer_idx = rho > 0.6

    for key in ['qi_itg', 'qe_itg', 'pfe_itg', 'qi_tem', 'qe_tem', 'pfe_tem']:
      npt.assert_allclose(
          result[key][inner_idx],
          model_output[key][inner_idx],
          rtol=1e-5,
          err_msg=f'Inner region should be unchanged for {key}',
      )
      self.assertTrue(
          bool(jnp.all(result[key][outer_idx] < model_output[key][outer_idx])),
          f'Outer region should be suppressed for {key}',
      )

  def test_etg_flux_unchanged_by_rotation_rule(self):
    """Tests that ETG flux is never modified by rotation rule."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = _make_mock_qualikiz_inputs(n_rho)
    geo = _make_mock_geometry(n_rho)

    for rotation_mode in [
        qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        qualikiz_based_transport_model.RotationMode.HALF_RADIUS,
    ]:
      result = qlknn_transport_model._maybe_apply_rotation_rule(
          model_output=model_output,
          qualikiz_inputs=qualikiz_inputs,
          rotation_mode=rotation_mode,
          shear_suppression_alpha=1.0,
          geo=geo,
      )
      npt.assert_array_equal(
          result['qe_etg'],
          model_output['qe_etg'],
          err_msg=(
              f'ETG flux should be unchanged for {rotation_mode}'
          ),
      )

  def test_zero_gamma_e_gives_no_suppression(self):
    """Tests that zero gamma_E_GB gives no suppression."""
    n_rho = 10
    model_output = _make_mock_model_output(n_rho)
    qualikiz_inputs = qualikiz_based_transport_model.QualikizInputs(
        Z_eff_face=jnp.ones(n_rho),
        lref_over_lti=jnp.ones(n_rho),
        lref_over_lte=jnp.ones(n_rho),
        lref_over_lne=jnp.ones(n_rho),
        lref_over_lni0=jnp.ones(n_rho),
        lref_over_lni1=jnp.ones(n_rho),
        q=jnp.linspace(1.0, 4.0, n_rho),
        smag=jnp.linspace(0.1, 2.0, n_rho),
        x=jnp.linspace(0.0, 1.0, n_rho),
        Ti_Te=jnp.ones(n_rho),
        log_nu_star_face=jnp.zeros(n_rho),
        normni=jnp.ones(n_rho),
        chiGB=jnp.ones(n_rho),
        Rmaj=jnp.array(6.2),
        Rmin=jnp.array(2.0),
        alpha=jnp.zeros(n_rho),
        epsilon=jnp.linspace(0.0, 0.3, n_rho),
        gamma_E_GB=jnp.zeros(n_rho),
        gamma_E_GB_poloidal_and_pressure=jnp.zeros(n_rho),
        gamma_E_GB_toroidal=jnp.zeros(n_rho),
        gamma_E_QLK=jnp.zeros(n_rho),
        mach_toroidal=jnp.zeros(n_rho),
    )
    geo = _make_mock_geometry(n_rho)

    result = qlknn_transport_model._maybe_apply_rotation_rule(
        model_output=model_output,
        qualikiz_inputs=qualikiz_inputs,
        rotation_mode=qualikiz_based_transport_model.RotationMode.FULL_RADIUS,
        shear_suppression_alpha=1.0,
        geo=geo,
    )

    for key in ['qi_itg', 'qe_itg', 'pfe_itg', 'qi_tem', 'qe_tem', 'pfe_tem']:
      npt.assert_allclose(
          result[key][1:],
          model_output[key][1:],
          rtol=1e-5,
          err_msg=f'Zero gamma_E_GB should give no suppression for {key}',
      )


if __name__ == '__main__':
  absltest.main()
