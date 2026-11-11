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

import dataclasses
from absl.testing import absltest
import chex
import jax.numpy as jnp
from torax._src.config import build_runtime_params
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry_model
from torax._src.geometry import runtime_params as geometry_runtime_params
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class GeometryModelTest(absltest.TestCase):

  def test_prescribed_geometry_model_initialize(self):
    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = (
        circular_geometry.CircularConfig(n_rho=4)
        .build_runtime_params(0.0)
        .geometry
    )
    geo_params = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo
    )
    runtime_params = dataclasses.replace(runtime_params, geometry=geo_params)
    model = geometry_model.PrescribedGeometryModel()

    geo_out, init_state = model.initialize(runtime_params)
    chex.assert_trees_all_close(geo_out, geo)
    chex.assert_trees_all_close(init_state.Phi_b, geo.Phi_b)
    chex.assert_trees_all_close(init_state.Phi_b_dot, 0.0)

  def test_prescribed_initialize_respects_initial_phi_b_dot(self):
    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = (
        circular_geometry.CircularConfig(n_rho=4)
        .build_runtime_params(0.0)
        .geometry
    )
    model = geometry_model.PrescribedGeometryModel()

    # calc_Phi_b_dot=True uses initial_Phi_b_dot
    geo_params_true = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo,
        calc_Phi_b_dot=True,
        initial_Phi_b_dot=jnp.asarray(1.75),
    )
    rp_true = dataclasses.replace(runtime_params, geometry=geo_params_true)
    geo_out, init_state = model.initialize(rp_true)
    chex.assert_trees_all_close(geo_out, geo)
    chex.assert_trees_all_close(init_state.Phi_b, geo.Phi_b)
    chex.assert_trees_all_close(init_state.Phi_b_dot, 1.75)

    # calc_Phi_b_dot=False ignores initial_Phi_b_dot and returns 0.0
    geo_params_false = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo,
        calc_Phi_b_dot=False,
        initial_Phi_b_dot=jnp.asarray(1.75),
    )
    rp_false = dataclasses.replace(runtime_params, geometry=geo_params_false)
    _, init_state_false = model.initialize(rp_false)
    chex.assert_trees_all_close(init_state_false.Phi_b_dot, 0.0)

  def test_prescribed_geometry_model_call(self):
    geo_base = (
        circular_geometry.CircularConfig(n_rho=4)
        .build_runtime_params(0.0)
        .geometry
    )
    geo_t = dataclasses.replace(
        geo_base, Phi_face=jnp.asarray(geo_base.Phi_face).at[-1].set(1.0)
    )
    geo_t_plus_dt = dataclasses.replace(
        geo_base, Phi_face=jnp.asarray(geo_base.Phi_face).at[-1].set(2.0)
    )
    model = geometry_model.PrescribedGeometryModel()

    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    geo_params_init = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo_t,
    )
    rp_init = dataclasses.replace(runtime_params, geometry=geo_params_init)
    _, initial_state = model.initialize(rp_init)

    # calc_Phi_b_dot = False
    geo_params_false = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo_t_plus_dt,
        calc_Phi_b_dot=False,
    )
    rp_false = dataclasses.replace(runtime_params, geometry=geo_params_false)
    geo_false, state_false = model(
        rp_false,
        initial_state,
        dt=0.5,
    )
    chex.assert_trees_all_close(geo_false, geo_t_plus_dt)
    chex.assert_trees_all_close(state_false.Phi_b, geo_t_plus_dt.Phi_b)
    chex.assert_trees_all_close(state_false.Phi_b_dot, 0.0)

    # calc_Phi_b_dot = True
    geo_params_true = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo_t_plus_dt,
        calc_Phi_b_dot=True,
    )
    rp_true = dataclasses.replace(runtime_params, geometry=geo_params_true)
    geo_true, state_true = model(
        rp_true,
        initial_state,
        dt=0.5,
    )
    chex.assert_trees_all_close(geo_true, geo_t_plus_dt)
    chex.assert_trees_all_close(state_true.Phi_b, geo_t_plus_dt.Phi_b)
    self.assertAlmostEqual(float(state_true.Phi_b_dot), 2.0)

  def test_initial_state_from_datatree(self):
    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = (
        circular_geometry.CircularConfig(n_rho=4)
        .build_runtime_params(0.0)
        .geometry
    )
    geo_params = geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo,
        initial_Phi_b_dot=jnp.asarray(1.5),
        calc_Phi_b_dot=True,
    )
    runtime_params = dataclasses.replace(runtime_params, geometry=geo_params)
    model = geometry_model.PrescribedGeometryModel()

    state = model.initial_state_from_datatree(None, runtime_params)
    chex.assert_trees_all_close(state.Phi_b, geo.Phi_b)
    chex.assert_trees_all_close(state.Phi_b_dot, 1.5)


if __name__ == '__main__':
  absltest.main()
