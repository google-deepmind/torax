# Copyright 2025 DeepMind Technologies Limited
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
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.edge import updaters
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_TARGET_EDGE_RATIO = 0.05
_INITIAL_EDGE_RATIO = 0.02
_INITIAL_AXIS_RATIO = 0.01
_UNCHANGED_RATIO = 0.015


class UpdateRuntimeParamsFromEdgeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['n_e_right_bc'] = 0.5
    config_dict['profile_conditions']['n_e_right_bc_is_fGW'] = True
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {
            'N': {0: _INITIAL_AXIS_RATIO, 1: _INITIAL_EDGE_RATIO},
            'Ne': {0: _UNCHANGED_RATIO, 1: _UNCHANGED_RATIO},
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    self.runtime_params = provider(t=0.0)
    self.edge_outputs = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(0.123),
        T_i_right_bc=jnp.array(0.456),
        n_e_right_bc=jnp.array(3.45e19),
        impurity_right_bc={'N': jnp.array(_TARGET_EDGE_RATIO)},
    )

  def _with_edge_flags(
      self,
      *,
      update_temperatures: bool = False,
      update_density: bool = False,
      update_impurities: bool = False,
  ):
    return dataclasses.replace(
        self.runtime_params,
        edge=edge_runtime_params.RuntimeParams(
            update_temperatures=update_temperatures,
            update_density=update_density,
            update_impurities=update_impurities,
        ),
    )

  def test_none_edge_outputs_is_noop(self):
    params = self._with_edge_flags(
        update_temperatures=True,
        update_density=True,
        update_impurities=True,
    )
    updated = updaters.update_runtime_params(params, None)
    self.assertIs(updated, params)

  def test_update_temperatures(self):
    params = self._with_edge_flags(update_temperatures=True)
    updated = updaters.update_runtime_params(params, self.edge_outputs)

    np.testing.assert_allclose(updated.profile_conditions.T_e_right_bc, 0.123)
    np.testing.assert_allclose(updated.profile_conditions.T_i_right_bc, 0.456)
    # Density boundary conditions remain unchanged.
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 0.5)
    self.assertTrue(updated.profile_conditions.n_e_right_bc_is_fGW)
    # Impurity conditions remain unchanged.
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    assert isinstance(
        params.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios['N'],
        params.plasma_composition.impurity.n_e_ratios['N'],
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios_face['N'],
        params.plasma_composition.impurity.n_e_ratios_face['N'],
    )

  def test_update_density(self):
    params = self._with_edge_flags(update_density=True)
    updated = updaters.update_runtime_params(params, self.edge_outputs)

    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 3.45e19)
    self.assertFalse(updated.profile_conditions.n_e_right_bc_is_fGW)
    # Temperature boundary conditions remain unchanged.
    np.testing.assert_allclose(
        updated.profile_conditions.T_e_right_bc,
        params.profile_conditions.T_e_right_bc,
    )
    np.testing.assert_allclose(
        updated.profile_conditions.T_i_right_bc,
        params.profile_conditions.T_i_right_bc,
    )
    # Impurity conditions remain unchanged.
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    assert isinstance(
        params.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios['N'],
        params.plasma_composition.impurity.n_e_ratios['N'],
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios_face['N'],
        params.plasma_composition.impurity.n_e_ratios_face['N'],
    )

  def test_update_impurities_scales_profile(self):
    params = self._with_edge_flags(update_impurities=True)
    initial_impurity_params = params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )

    updated = updaters.update_runtime_params(params, self.edge_outputs)
    updated_impurity_params = updated.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )

    scaling_factor = _TARGET_EDGE_RATIO / _INITIAL_EDGE_RATIO
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['N'],
        initial_impurity_params.n_e_ratios['N'] * scaling_factor,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['N'],
        initial_impurity_params.n_e_ratios_face['N'] * scaling_factor,
        rtol=1e-5,
    )
    # 'Ne' is not in edge_outputs.impurity_right_bc, so it remains unchanged.
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['Ne'],
        initial_impurity_params.n_e_ratios['Ne'],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['Ne'],
        initial_impurity_params.n_e_ratios_face['Ne'],
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
