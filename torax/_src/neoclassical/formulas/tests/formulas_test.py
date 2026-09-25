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
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical.formulas import formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
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
            'Z_eff': 2.0,
        },
        'geometry': {
            'geometry_type': 'chease',
            'Ip_from_parameters': False,
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })

    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params,
        self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()  # pyrefly: ignore[bad-argument-type]
    )
    self.nu_e_star = formulas.calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=self.geo,
        n_e=self.core_profiles.n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        T_e=self.core_profiles.T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = formulas.calculate_f_trap(self.geo)

  def test_calculate_f_trap_positive_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_negative_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(-0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_gradient_on_axis(self):
    grad_fn = jax.grad(
        lambda geo: jnp.sum(formulas.calculate_f_trap(geo)),
        allow_int=True,
    )
    grad_geo = grad_fn(self.geo)

    for leaf in jax.tree_util.tree_leaves(grad_geo):
      if isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
          leaf.dtype, jnp.inexact
      ):
        chex.assert_tree_all_finite(leaf)


if __name__ == '__main__':
  absltest.main()
