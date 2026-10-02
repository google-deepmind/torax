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

"""Tests for the gyaradax transport models."""

from collections.abc import Mapping
import dataclasses
import math
import os
import pickle
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

try:
  import gyaradax  # pylint: disable=unused-import

  _GYARADAX_AVAILABLE = True
except ImportError:
  _GYARADAX_AVAILABLE = False


if _GYARADAX_AVAILABLE:
  import jax.numpy as jnp
  from torax._src.config import build_runtime_params
  from torax._src.core_profiles import initialization
  from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
  from torax._src.sources import source_profile_builders
  from torax._src.test_utils import default_configs
  from torax._src.torax_pydantic import model_config
  from torax._src.transport_model import gyaradax_base
  from torax._src.transport_model import gyaradax_diagnostics as diag_lib
  from torax._src.transport_model import gyaradax_nl_transport_model
  from torax._src.transport_model import gyaradax_normalization as gb_norm
  from torax._src.transport_model import gyaradax_ql_transport_model



_MODEL_KEY = 'gyaradax'


def _get_config_and_model_inputs(transport: Mapping[str, Any]):
  """Returns the torax config and model inputs for testing."""
  config = default_configs.get_default_config_dict()
  config['transport'] = {'core_transport_models': {_MODEL_KEY: transport}}
  torax_config = model_config.ToraxConfig.from_dict(config)
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=torax_config.numerics.t_initial)
  geo = torax_config.geometry.build_provider(t=torax_config.numerics.t_initial)
  core_profiles = initialization.initial_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
  )
  source_profiles = source_profile_builders.build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      explicit=True,
  )
  pedestal_model = torax_config.pedestal.build_pedestal_model()
  pedestal_model_outputs = pedestal_model(
      runtime_params,
      geo,
      core_profiles,
      source_profiles,
      pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
  )
  del pedestal_model_outputs
  two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
  return torax_config, (
      runtime_params,
      geo,
      core_profiles,
      two_point_mask,
  )


if _GYARADAX_AVAILABLE:

  @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
  class FakeGyaradaxTransportModel(
      gyaradax_ql_transport_model.GyaradaxQLTransportModel
  ):
    """Constant-flux stub exercising the vmap/interp/conversion machinery."""

    qi_gb: float = 1.0
    qe_gb: float = 0.5
    pfe_gb: float = 0.0
    em: bool = False

    def _per_radius(self, params, geom):
      del params, geom
      return (
          jnp.asarray(self.qi_gb),
          jnp.asarray(self.qe_gb),
          jnp.asarray(self.pfe_gb),
          {},
      )




@absltest.skipUnless(_GYARADAX_AVAILABLE, 'gyaradax is not installed')
class GyaradaxNormalizationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _, model_inputs = _get_config_and_model_inputs({'model_name': 'prescribed'})
    _, self.geo, self.core_profiles, _ = model_inputs

  def test_conversion_identity(self):
    factor = gb_norm.gb_flux_conversion_factor(
        gb_norm.GYARADAX, gb_norm.GYARADAX, self.geo
    )
    np.testing.assert_allclose(factor, 1.0)

  def test_gyaradax_to_torax_ql_is_two_sqrt_two(self):
    # same L_ref and T_chi: only GKW's sqrt(2) velocity choice contributes
    factor = gb_norm.gb_flux_conversion_factor(
        gb_norm.GYARADAX, gb_norm.TORAX_QL, self.geo
    )
    np.testing.assert_allclose(factor, 2.0 * math.sqrt(2.0))

  def test_torax_ql_to_qlknn_hyper_factor(self):
    factor = gb_norm.gb_flux_conversion_factor(
        gb_norm.TORAX_QL, gb_norm.QLKNN_HYPER, self.geo, self.core_profiles
    )
    t_i = self.core_profiles.T_i.face_value()
    t_e = self.core_profiles.T_e.face_value()
    expected = (self.geo.a_minor / self.geo.R_major) ** 2 * (t_i / t_e) ** 1.5
    np.testing.assert_allclose(factor, expected, rtol=1e-12)



  def test_build_quasilinear_inputs_shapes(self):
    ql_inputs = gb_norm.build_quasilinear_inputs(self.core_profiles, self.geo)
    expected_shape = self.geo.rho_face_norm.shape
    self.assertEqual(ql_inputs.chiGB.shape, expected_shape)
    self.assertEqual(ql_inputs.lref_over_lti.shape, expected_shape)
    self.assertEqual(ql_inputs.lref_over_lne.shape, expected_shape)


@absltest.skipUnless(_GYARADAX_AVAILABLE, 'gyaradax is not installed')
class GyaradaxTransportModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _, model_inputs = _get_config_and_model_inputs(
        {'model_name': 'gyaradax-ql'}
    )
    (
        self.runtime_params,
        self.geo,
        self.core_profiles,
        self.two_point_mask,
    ) = model_inputs

  def test_face_indices_for_radii(self):
    idx = gyaradax_base.face_indices_for_radii(
        self.geo, (0.0, 0.5, 1.0)
    )
    rho_face = np.asarray(self.geo.rho_face_norm)
    self.assertEqual(int(idx[0]), 0)
    self.assertEqual(int(idx[2]), rho_face.shape[0] - 1)
    self.assertEqual(int(idx[1]), int(np.argmin(np.abs(rho_face - 0.5))))


  def test_gkparams_for_radius_clips_and_defaults(self):
    model = FakeGyaradaxTransportModel()
    ql_inputs = gb_norm.build_quasilinear_inputs(self.core_profiles, self.geo)
    idx = self.geo.rho_face_norm.shape[0] // 2
    params = gyaradax_base.gkparams_for_radius(
        idx, ql_inputs, self.core_profiles, self.geo, model
    )
    self.assertBetween(float(params.rlt), 0.0, 30.0)
    self.assertBetween(float(params.rln), -15.0, 15.0)
    self.assertBetween(float(params.q), 0.5, 10.0)
    self.assertBetween(float(params.shat), -3.0, 6.0)
    self.assertBetween(float(params.eps), 0.02, 0.5)
    self.assertEqual(float(params.beta), 0.0)
    self.assertFalse(params.nlapar)
    self.assertTrue(params.adiabatic_electrons)
    self.assertFalse(params.non_linear)
    self.assertTrue(params.disable_per_ky_norm)


  def test_call_implementation_shapes_and_gb_plumbing(self):
    # acceptance check: qi=1 GKW-GB -> chi_i = 2*sqrt(2) * chiGB / (R/L_Ti)
    model = FakeGyaradaxTransportModel(qi_gb=1.0, qe_gb=0.5, pfe_gb=0.0)
    core_transport = model.call_implementation(
        self.runtime_params.transport.core_transport_model_params[_MODEL_KEY],
        self.runtime_params,
        self.geo,
        self.core_profiles,
        self.two_point_mask,
    )
    expected_shape = self.geo.rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

    ql_inputs = gb_norm.build_quasilinear_inputs(self.core_profiles, self.geo)
    expected_chi_ion = (
        2.0 * math.sqrt(2.0) * ql_inputs.chiGB / ql_inputs.lref_over_lti
    )
    # skip the magnetic axis where R/L_Ti -> 0
    np.testing.assert_allclose(
        np.asarray(core_transport.chi_face_ion)[1:],
        np.asarray(expected_chi_ion)[1:],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(core_transport.chi_face_el)[1:],
        0.5
        * np.asarray(
            2.0 * math.sqrt(2.0) * ql_inputs.chiGB / ql_inputs.lref_over_lte
        )[1:],
        rtol=1e-6,
    )


@absltest.skipUnless(_GYARADAX_AVAILABLE, 'gyaradax is not installed')
class GyaradaxQLConfigTest(parameterized.TestCase):

  def test_config_roundtrip(self):
    torax_config, _ = _get_config_and_model_inputs({
        'model_name': 'gyaradax-ql',
        'rho_match': (0.3, 0.6),
        'n_steps_linear': 400,
        'early_stop': False,
        'nkx': 43,
        'nky': 16,
    })
    model = torax_config.transport.core_transport_models[
        _MODEL_KEY
    ].build_transport_model()
    self.assertIsInstance(
        model, gyaradax_ql_transport_model.GyaradaxQLTransportModel
    )
    self.assertEqual(model.rho_match, (0.3, 0.6))
    self.assertEqual(model.n_steps_linear, 400)
    self.assertFalse(model.early_stop)
    self.assertFalse(model.em)
    self.assertEqual(model.backend, 'jax')

  def test_runtime_params_carry_particle_knobs(self):
    torax_config, _ = _get_config_and_model_inputs(
        {'model_name': 'gyaradax-ql', 'An_min': 0.1}
    )
    runtime_params = torax_config.transport.core_transport_models[
        _MODEL_KEY
    ].build_runtime_params(0.0)
    self.assertIsInstance(
        runtime_params, gyaradax_ql_transport_model.RuntimeParams
    )
    self.assertEqual(float(runtime_params.An_min), 0.1)

  def test_grid_adopted_from_calibration_head(self):
    import pickle
    import tempfile

    head_grid = {
        'nvpar': 48,
        'nmu': 12,
        'ns': 32,
        'nkx': 85,
        'nky': 32,
        'ikxspace': 3,
        'krhomax': 1.1,
    }
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
      pickle.dump({'scalar': 2.0, 'grid': head_grid}, f)
      head_path = f.name
    config = gyaradax_ql_transport_model.GyaradaxQLConfig.from_dict(
        {'model_name': 'gyaradax-ql', 'cn_calibration_path': head_path}
    )
    model = config.build_transport_model()
    for key, value in head_grid.items():
      self.assertEqual(getattr(model, key), value)
    # explicit config wins over the head's grid
    config = gyaradax_ql_transport_model.GyaradaxQLConfig.from_dict({
        'model_name': 'gyaradax-ql',
        'cn_calibration_path': head_path,
        'nky': 48,
    })
    self.assertEqual(config.build_transport_model().nky, 48)

  def test_grid_below_floor_is_raised(self):
    # the perpendicular resolution floor overrides an explicit sub-floor grid
    config = gyaradax_ql_transport_model.GyaradaxQLConfig.from_dict({
        'model_name': 'gyaradax-ql',
        'nvpar': 32,
        'nmu': 8,
        'ns': 16,
        'nkx': 43,
        'nky': 16,
        'ikxspace': 5,
        'krhomax': 1.4,
    })
    with self.assertWarnsRegex(RuntimeWarning, 'resolution floor'):
      model = config.build_transport_model()
    self.assertEqual(model.nkx, 85)
    self.assertEqual(model.nky, 32)


  def test_grid_underspecified_raises(self):
    # no head (cn path None) and no explicit grid: there is nothing to adopt
    config = gyaradax_ql_transport_model.GyaradaxQLConfig.from_dict(
        {'model_name': 'gyaradax-ql', 'cn_calibration_path': None}
    )
    with self.assertRaisesRegex(ValueError, 'grid underspecified'):
      config.build_transport_model()

  def test_fast_ion_stabilization_raises(self):
    config = gyaradax_ql_transport_model.GyaradaxQLConfig.from_dict(
        {'model_name': 'gyaradax-ql', 'fast_ion_stabilization': 1.0}
    )
    with self.assertRaises(NotImplementedError):
      config.build_transport_model()








@absltest.skipUnless(_GYARADAX_AVAILABLE, 'gyaradax is not installed')
class GyaradaxDiagnosticsTest(parameterized.TestCase):
  """Host-side diagnostics sink and the jsonl / latent-npz it writes."""

  def setUp(self):
    super().setUp()
    diag_lib.reset_sinks()
    _, model_inputs = _get_config_and_model_inputs(
        {'model_name': 'gyaradax-ql'}
    )
    (
        self.runtime_params,
        self.geo,
        self.core_profiles,
        self.two_point_mask,
    ) = model_inputs

  def _call(self, model):
    return model.call_implementation(
        self.runtime_params.transport.core_transport_model_params[_MODEL_KEY],
        self.runtime_params,
        self.geo,
        self.core_profiles,
        self.two_point_mask,
    )


  def test_ql_rows_carry_local_parameters_and_raw_fluxes(self):
    path = os.path.join(self.create_tempdir().full_path, 'diag.jsonl')
    model = FakeGyaradaxTransportModel(
        rho_match=(0.3, 0.6), qi_gb=3.0, qe_gb=1.5, diagnostics_path=path
    )
    self._call(model)
    rows = diag_lib.load_jsonl(path)
    self.assertLen(rows, 2)
    self.assertEqual([r['radius'] for r in rows], [0, 1])
    self.assertEqual([r['call'] for r in rows], [0, 0])
    for row in rows:
      # fluxes are recorded before the gyroBohm unit conversion
      self.assertAlmostEqual(row['qi_gb'], 3.0)
      self.assertAlmostEqual(row['qe_gb'], 1.5)
      for key in ('rlt', 'rln', 'q', 'shat', 'eps', 'beta', 'rho_face'):
        self.assertIsInstance(row[key], float)
      self.assertGreaterEqual(row['wall_s'], 0.0)







if __name__ == '__main__':
  absltest.main()
