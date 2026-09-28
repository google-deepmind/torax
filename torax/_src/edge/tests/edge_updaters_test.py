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


from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import updaters
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class UpdateRuntimeParamsFromEdgeTest(parameterized.TestCase):

  def test_update_impurities_scales_profile(self):
    _TARGET_EDGE_RATIO = 0.05
    _INITIAL_EDGE_RATIO = 0.02
    _INITIAL_AXIS_RATIO = 0.01
    _UNCHANGED_RATIO = 0.015
    config_dict = default_configs.get_default_config_dict()
    # Set impurity mode to n_e_ratios and define profiles for 'N' (updated)
    # and 'Ne' (unchanged).
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {
            'N': {0: _INITIAL_AXIS_RATIO, 1: _INITIAL_EDGE_RATIO},
            'Ne': {0: _UNCHANGED_RATIO, 1: _UNCHANGED_RATIO},
        },
    }
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'update_impurities': True,
        'use_enrichment_model': False,
        'enrichment_factor': {'N': 2.0, 'Ne': 2.0},
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {'Ne': 0.01},
        'T_e_target': 1.0,
        'connection_length_target': 1.0,
        'connection_length_divertor': 1.0,
        'toroidal_flux_expansion': 1.0,
        'angle_of_incidence_target': 1.0,
        'diverted': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.impurity_right_bc = {'N': jnp.array(_TARGET_EDGE_RATIO)}
    edge_outputs.T_e_right_bc = 1.0
    edge_outputs.T_i_right_bc = 1.0

    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios_N = initial_impurity_params.n_e_ratios['N']
    initial_n_e_ratios_face_N = initial_impurity_params.n_e_ratios_face['N']
    initial_n_e_ratios_Ne = initial_impurity_params.n_e_ratios['Ne']
    initial_n_e_ratios_face_Ne = initial_impurity_params.n_e_ratios_face['Ne']

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )

    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )

    scaling_factor = _TARGET_EDGE_RATIO / _INITIAL_EDGE_RATIO

    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['N'],
        initial_n_e_ratios_N * scaling_factor,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['N'],
        initial_n_e_ratios_face_N * scaling_factor,
        rtol=1e-5,
    )
    # 'Ne' is not in edge_outputs.impurity_right_bc, so it should remain
    # unchanged.
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['Ne'],
        initial_n_e_ratios_Ne,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['Ne'],
        initial_n_e_ratios_face_Ne,
        rtol=1e-5,
    )

  def test_update_temperatures(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {'N': 0.01},
    }
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'update_impurities': False,
        'update_temperatures': True,
        'use_enrichment_model': False,
        'fixed_impurity_concentrations': {'N': 0.01},
        'enrichment_factor': {'N': 1.0},
        'connection_length_target': 1.0,
        'connection_length_divertor': 1.0,
        'toroidal_flux_expansion': 1.0,
        'angle_of_incidence_target': 1.0,
        'diverted': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.T_e_right_bc = jnp.array(0.123)
    edge_outputs.T_i_right_bc = jnp.array(0.456)
    edge_outputs.impurity_right_bc = {}

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )

    np.testing.assert_allclose(
        updated_runtime_params.profile_conditions.T_e_right_bc, 0.123
    )
    np.testing.assert_allclose(
        updated_runtime_params.profile_conditions.T_i_right_bc, 0.456
    )


if __name__ == '__main__':
  absltest.main()
