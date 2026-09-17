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
"""Tests for the GACODE NEO surrogate neoclassical transport model."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import neo_surrogate
from torax._src.torax_pydantic import model_config

_N_RHO = 15


class NeoSurrogateTransportTest(parameterized.TestCase):

  def _get_runtime_params_geo_and_core_profiles(
      self,
  ) -> tuple[
      runtime_params_lib.RuntimeParams, geometry.Geometry, state.CoreProfiles
  ]:
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'Ip': 15e6,
            'current_profile_nu': 3,
            'n_e_nbar_is_fGW': True,
            'normalize_n_e_to_nbar': True,
            'nbar': 0.85,
            'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
        },
        'numerics': {},
        'plasma_composition': {
            'Z_eff': 1.5,
        },
        'geometry': {
            'geometry_type': 'chease',
            'Ip_from_parameters': False,
            'n_rho': _N_RHO,
        },
        'neoclassical': {
            'transport': {'model_name': 'neo_surrogate'},
            'bootstrap_current': {'model_name': 'neo_surrogate'},
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )

    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    return runtime_params, geo, core_profiles

  def test_subclass_conformance(self):
    self.assertTrue(
        issubclass(
            neo_surrogate.NeoSurrogateTransportModel,
            base.NeoclassicalTransportModel,
        )
    )
    model = neo_surrogate.NeoSurrogateTransportModel()
    self.assertIsInstance(model, base.NeoclassicalTransportModel)

  def test_config_builder(self):
    cfg = neo_surrogate.NeoSurrogateModelConfig()
    self.assertEqual(cfg.model_name, 'neo_surrogate')
    model = cfg.build_model()
    self.assertIsInstance(model, neo_surrogate.NeoSurrogateTransportModel)
    params = cfg.build_runtime_params()
    self.assertIsInstance(params, neo_surrogate.RuntimeParams)

  def test_transport_evaluation_shapes_and_positivity(self):
    runtime_params, geo, core_profiles = (
        self._get_runtime_params_geo_and_core_profiles()
    )
    model = neo_surrogate.NeoSurrogateTransportModel()

    res = model(runtime_params, geo, core_profiles)
    n_face = _N_RHO + 1

    self.assertEqual(res.chi_neo_i.shape, (n_face,))
    self.assertEqual(res.chi_neo_e.shape, (n_face,))
    self.assertEqual(res.D_neo_e.shape, (n_face,))
    self.assertEqual(res.V_neo_e.shape, (n_face,))
    self.assertEqual(res.V_neo_ware_e.shape, (n_face,))

    # Strict finiteness
    self.assertTrue(jnp.all(jnp.isfinite(res.chi_neo_i)))
    self.assertTrue(jnp.all(jnp.isfinite(res.chi_neo_e)))
    self.assertTrue(jnp.all(jnp.isfinite(res.D_neo_e)))
    self.assertTrue(jnp.all(jnp.isfinite(res.V_neo_e)))
    self.assertTrue(jnp.all(jnp.isfinite(res.V_neo_ware_e)))

    # Strict positivity for diffusivities
    self.assertTrue(jnp.all(res.chi_neo_i >= 0.0))
    self.assertTrue(jnp.all(res.chi_neo_e >= 0.0))
    self.assertTrue(jnp.all(res.D_neo_e >= 0.0))

  @parameterized.parameters(
      (6.0, 12.011),  # Carbon
      (74.0, 183.84),  # Tungsten
      (2.0, 4.0026),  # Helium
  )
  def test_impurity_transport(self, z_imp, a_imp):
    runtime_params, geo, core_profiles = (
        self._get_runtime_params_geo_and_core_profiles()
    )
    model = neo_surrogate.NeoSurrogateTransportModel()

    d_imp, v_imp = model.calculate_impurity_transport(
        runtime_params, geo, core_profiles, Z_imp=z_imp, A_imp=a_imp
    )
    n_face = _N_RHO + 1
    self.assertEqual(d_imp.shape, (n_face,))
    self.assertEqual(v_imp.shape, (n_face,))
    self.assertTrue(jnp.all(jnp.isfinite(d_imp)))
    self.assertTrue(jnp.all(jnp.isfinite(v_imp)))
    self.assertTrue(jnp.all(d_imp >= 0.0))

  def test_jit_compilation(self):
    runtime_params, geo, core_profiles = (
        self._get_runtime_params_geo_and_core_profiles()
    )
    model = neo_surrogate.NeoSurrogateTransportModel()

    @jax.jit
    def eval_fn(profiles):
      return model(runtime_params, geo, profiles)

    res = eval_fn(core_profiles)
    self.assertTrue(jnp.all(jnp.isfinite(res.chi_neo_i)))


if __name__ == '__main__':
  absltest.main()
