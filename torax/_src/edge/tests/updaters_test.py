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
import chex
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import extended_lengyel_model
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import updaters
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class UpdateRuntimeParamsFromEdgeTest(parameterized.TestCase):

  def test_update_impurities_scales_profile(self):
    _ENRICHMENT_FACTOR = 2.0
    _OUTPUT_CONCENTRATION = 0.1
    _INITIAL_EDGE_RATIO = 0.02
    _INITIAL_AXIS_RATIO = 0.01
    config_dict = default_configs.get_default_config_dict()
    # Set impurity mode to n_e_ratios and define a profile
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {'N': {0: _INITIAL_AXIS_RATIO, 1: _INITIAL_EDGE_RATIO}},
    }
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    # Set up edge model config
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'update_impurities': True,
        'use_enrichment_model': False,
        'enrichment_factor': {'N': _ENRICHMENT_FACTOR},
        'seed_impurity_weights': {'N': 1.0},
        # Dummy values for other required fields.
        'target_electron_temp': 1.0,
        'parallel_connection_length': 1.0,
        'divertor_parallel_length': 1.0,
        'toroidal_flux_expansion': 1.0,
        'target_angle_of_incidence': 1.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.seed_impurity_concentrations = {
        'N': jnp.array(_OUTPUT_CONCENTRATION)
    }
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value for tracing.

    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios = initial_impurity_params.n_e_ratios['N']

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )

    updated_impurity_params = updated_runtime_params.plasma_composition.impurity

    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    updated_n_e_ratios = updated_impurity_params.n_e_ratios['N']

    # Expected scaling logic:
    conc_lcfs = _OUTPUT_CONCENTRATION / _ENRICHMENT_FACTOR
    scaling_factor = conc_lcfs / _INITIAL_EDGE_RATIO

    initial_n_e_ratios_face = initial_impurity_params.n_e_ratios_face['N']
    updated_n_e_ratios_face = updated_impurity_params.n_e_ratios_face['N']

    np.testing.assert_allclose(
        updated_n_e_ratios, initial_n_e_ratios * scaling_factor, rtol=1e-5
    )
    np.testing.assert_allclose(
        updated_n_e_ratios_face,
        initial_n_e_ratios_face * scaling_factor,
        rtol=1e-5,
    )


class UpdateFixedImpuritiesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._ENRICHMENT_FACTOR = 2.0
    self._INITIAL_EDGE_RATIO = 0.02
    self._INITIAL_AXIS_RATIO = 0.01
    self._EDGE_CONCENTRATION = 0.05
    self.config_dict = default_configs.get_default_config_dict()
    # Common config parts
    self.config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {
            'N': {0: self._INITIAL_AXIS_RATIO, 1: self._INITIAL_EDGE_RATIO},
            'Ne': {0: 0.0, 1: 0.0},  # Dummy values in setup
        },
    }
    self.config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    # Base edge config, to be modified in each test
    self.config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'update_impurities': True,
        'update_temperatures': False,
        'use_enrichment_model': False,
        'enrichment_factor': {
            'N': self._ENRICHMENT_FACTOR,
            'Ne': self._ENRICHMENT_FACTOR,
        },
        # Dummy values
        'parallel_connection_length': 1.0,
        'divertor_parallel_length': 1.0,
        'toroidal_flux_expansion': 1.0,
        'target_angle_of_incidence': 1.0,
    }

  def test_update_fixed_impurities_edge_truth_forward_mode(self):
    self.config_dict['edge']['impurity_sot'] = 'edge'
    self.config_dict['edge']['computation_mode'] = 'forward'
    self.config_dict['edge']['seed_impurity_weights'] = None
    self.config_dict['edge']['target_electron_temp'] = None
    self.config_dict['edge']['fixed_impurity_concentrations'] = {
        'N': self._EDGE_CONCENTRATION,
        'Ne': 0.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value
    edge_outputs.seed_impurity_concentrations = {}  # No seeded impurity update
    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios = initial_impurity_params.n_e_ratios['N']
    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    updated_n_e_ratios = updated_impurity_params.n_e_ratios['N']
    # Expected scaling logic:
    conc_lcfs = self._EDGE_CONCENTRATION / self._ENRICHMENT_FACTOR
    scaling_factor = conc_lcfs / self._INITIAL_EDGE_RATIO
    np.testing.assert_allclose(
        updated_n_e_ratios, initial_n_e_ratios * scaling_factor, rtol=1e-5
    )

  def test_update_fixed_impurities_edge_truth_inverse_mode(self):
    self.config_dict['edge']['impurity_sot'] = 'edge'
    self.config_dict['edge']['computation_mode'] = 'inverse'
    self.config_dict['edge']['seed_impurity_weights'] = {'Ne': 1.0}
    self.config_dict['edge']['target_electron_temp'] = 1.0
    self.config_dict['edge']['fixed_impurity_concentrations'] = {
        'N': self._EDGE_CONCENTRATION
    }
    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value
    edge_outputs.seed_impurity_concentrations = {}  # No seeded impurity update
    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios = initial_impurity_params.n_e_ratios['N']
    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    updated_n_e_ratios = updated_impurity_params.n_e_ratios['N']
    # Expected scaling logic:
    conc_lcfs = self._EDGE_CONCENTRATION / self._ENRICHMENT_FACTOR
    scaling_factor = conc_lcfs / self._INITIAL_EDGE_RATIO
    np.testing.assert_allclose(
        updated_n_e_ratios, initial_n_e_ratios * scaling_factor, rtol=1e-5
    )

  def test_update_fixed_impurities_core_truth_forward_mode(self):
    self.config_dict['edge']['impurity_sot'] = 'core'
    self.config_dict['edge']['computation_mode'] = 'forward'
    self.config_dict['edge']['seed_impurity_weights'] = None
    self.config_dict['edge']['target_electron_temp'] = None
    self.config_dict['edge']['fixed_impurity_concentrations'] = {
        'N': self._EDGE_CONCENTRATION,
        'Ne': 0.0,
    }  # Dummy Edge value (should be ignored for core update)
    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.seed_impurity_concentrations = {}
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value for tracing.
    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    initial_impurity_params = runtime_params.plasma_composition.impurity
    # Should be identical
    chex.assert_trees_all_equal(
        updated_impurity_params, initial_impurity_params
    )

  def test_update_fixed_impurities_core_truth_inverse_mode(self):
    self.config_dict['edge']['impurity_sot'] = 'core'
    self.config_dict['edge']['computation_mode'] = 'inverse'
    self.config_dict['edge']['seed_impurity_weights'] = {'Ne': 1.0}
    self.config_dict['edge']['target_electron_temp'] = 1.0
    self.config_dict['edge']['fixed_impurity_concentrations'] = {
        'N': self._EDGE_CONCENTRATION
    }  # Dummy Edge value (should be ignored for core update)
    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.seed_impurity_concentrations = {}
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value for tracing.
    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    initial_impurity_params = runtime_params.plasma_composition.impurity
    # Should be identical
    chex.assert_trees_all_equal(
        updated_impurity_params, initial_impurity_params
    )


class UpdateImpuritiesWithEnrichmentModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._CALCULATED_ENRICHMENT = 3.0
    self._OUTPUT_CONCENTRATION = 0.1
    self._INITIAL_EDGE_RATIO = 0.02
    self._INITIAL_AXIS_RATIO = 0.01
    self.config_dict = default_configs.get_default_config_dict()
    self.config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {
            'N': {0: self._INITIAL_AXIS_RATIO, 1: self._INITIAL_EDGE_RATIO}
        },
    }
    self.config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    # Base edge config, to be modified in each test
    self.config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'update_impurities': True,
        'update_temperatures': False,  # disable to simplify test
        'use_enrichment_model': True,
        'seed_impurity_weights': {'N': 1.0},
        # Dummy values for other required fields.
        'target_electron_temp': 1.0,
        'parallel_connection_length': 1.0,
        'divertor_parallel_length': 1.0,
        'toroidal_flux_expansion': 1.0,
        'target_angle_of_incidence': 1.0,
    }

  def test_update_impurities_scales_profile_with_enrichment_model(self):
    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    # Mock edge_outputs
    edge_outputs = mock.MagicMock(
        spec=extended_lengyel_standalone.ExtendedLengyelOutputs
    )
    edge_outputs.seed_impurity_concentrations = {
        'N': jnp.array(self._OUTPUT_CONCENTRATION)
    }
    edge_outputs.calculated_enrichment = {
        'N': jnp.array(self._CALCULATED_ENRICHMENT)
    }
    edge_outputs.separatrix_electron_temp = 1.0  # Dummy value for tracing.

    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios = initial_impurity_params.n_e_ratios['N']

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )

    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    updated_n_e_ratios = updated_impurity_params.n_e_ratios['N']

    # Expected scaling logic:
    conc_lcfs = self._OUTPUT_CONCENTRATION / self._CALCULATED_ENRICHMENT
    scaling_factor = conc_lcfs / self._INITIAL_EDGE_RATIO

    np.testing.assert_allclose(
        updated_n_e_ratios, initial_n_e_ratios * scaling_factor, rtol=1e-5
    )

  @parameterized.named_parameters(
      ('core_sot_model_on', 'core', True),
      ('edge_sot_model_on', 'edge', False),
  )
  def test_updates_enrichment_factor_conditionally_when_use_enrichment_model_true(
      self, impurity_sot, should_update
  ):
    self.config_dict['edge']['impurity_sot'] = impurity_sot
    self.config_dict['edge']['use_enrichment_model'] = True

    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
    initial_enrichment_factor = runtime_params.edge.enrichment_factor

    edge_outputs = mock.MagicMock(
        spec=extended_lengyel_standalone.ExtendedLengyelOutputs
    )
    edge_outputs.calculated_enrichment = {
        'N': jnp.array(self._CALCULATED_ENRICHMENT)
    }
    edge_outputs.seed_impurity_concentrations = {}
    edge_outputs.separatrix_electron_temp = 1.0

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    assert isinstance(
        updated_runtime_params.edge, extended_lengyel_model.RuntimeParams
    )
    updated_enrichment_factor = updated_runtime_params.edge.enrichment_factor

    if should_update:
      # It should be updated to the value from edge_outputs
      np.testing.assert_allclose(
          updated_enrichment_factor['N'], self._CALCULATED_ENRICHMENT
      )
      # And it should be different from the initial value
      self.assertNotEqual(
          initial_enrichment_factor['N'], updated_enrichment_factor['N']
      )
    else:
      # It should not have been updated
      np.testing.assert_allclose(
          updated_enrichment_factor['N'], initial_enrichment_factor['N']
      )

  @parameterized.named_parameters(
      ('core_sot_model_off', 'core'),
      ('edge_sot_model_off', 'edge'),
  )
  def test_does_not_update_enrichment_factor_when_use_enrichment_model_false(
      self, impurity_sot
  ):
    self.config_dict['edge']['impurity_sot'] = impurity_sot
    self.config_dict['edge']['use_enrichment_model'] = False
    self.config_dict['edge']['enrichment_factor'] = {'N': 5.0}

    torax_config = model_config.ToraxConfig.from_dict(self.config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
    initial_enrichment_factor = runtime_params.edge.enrichment_factor

    edge_outputs = mock.MagicMock(
        spec=extended_lengyel_standalone.ExtendedLengyelOutputs
    )
    edge_outputs.calculated_enrichment = {
        'N': jnp.array(self._CALCULATED_ENRICHMENT)
    }
    edge_outputs.seed_impurity_concentrations = {}
    edge_outputs.separatrix_electron_temp = 1.0

    updated_runtime_params = updaters.update_runtime_params(
        runtime_params, edge_outputs
    )
    assert isinstance(
        updated_runtime_params.edge, extended_lengyel_model.RuntimeParams
    )
    updated_enrichment_factor = updated_runtime_params.edge.enrichment_factor

    # It should not have been updated
    np.testing.assert_allclose(
        updated_enrichment_factor['N'], initial_enrichment_factor['N']
    )


if __name__ == '__main__':
  absltest.main()
