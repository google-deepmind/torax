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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.core_profiles import (
    runtime_params as core_profile_runtime_params,
)
from torax._src.geometry import circular_geometry
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class SourceModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()

  def test_computing_source_profiles_works_with_all_defaults(self):
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    source_states = source_models.initial_state(runtime_params)
    _, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, core_profiles
        )
    )
    explicit_source_profiles, source_states = (
        source_profile_builders.build_source_profiles(
            runtime_params,
            geo,
            core_profiles,
            source_models,
            source_states=source_states,
            explicit=True,
        )
    )
    source_profile_builders.build_source_profiles(
        runtime_params,
        geo,
        core_profiles,
        source_models,
        source_states=source_states,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        bootstrap_current=bootstrap_current,
    )

  def _build_test_env(
      self,
      is_explicit: bool = True,
  ):
    """Constructs real Torax objects with ecrh, icrh, ohmic, and impurity."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry']['n_rho'] = 4
    config_dict['numerics']['enable_fast_ions'] = True
    config_dict['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'He3': 0.03},
        },
    }
    config_dict['sources']['ecrh'] = {
        'is_explicit': is_explicit,
        'P_total': 5e6,
    }
    config_dict['sources']['icrh'] = {
        'model_name': 'scaled_profile',
        'minority_species': 'He3',
        'is_explicit': is_explicit,
        'P_total': 10e6,
        'heat_profile_ion': {0: {0: 1.0, 1: 1.0}},
        'heat_profile_electron': {0: {0: 1.0, 1: 1.0}},
    }
    config_dict['sources']['impurity_radiation'] = {
        'model_name': 'P_in_scaled_flat_profile',
        'fraction_P_heating': 0.2,
        'is_explicit': False,
    }
    config_dict['sources']['ohmic'] = {
        'is_explicit': False,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_states = source_models.initial_state(runtime_params)
    conductivity, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, core_profiles
        )
    )
    return (
        source_models,
        runtime_params,
        geo,
        core_profiles,
        source_states,
        conductivity,
        bootstrap_current,
    )

  @parameterized.parameters(True, False)
  def test_build_basic_sources_explicit_and_implicit(
      self, is_explicit: bool
  ):
    (
        source_models,
        runtime_params,
        geo,
        core_profiles,
        source_states,
        _,
        _,
    ) = self._build_test_env(is_explicit=is_explicit)

    explicit_outputs = None
    if not is_explicit:
      explicit_outputs, _ = source_profile_builders.build_basic_sources(
          runtime_params,
          geo,
          core_profiles,
          source_models,
          source_states=source_states,
          explicit=True,
      )

    pfi = (
        core_profile_runtime_params.PrescribedFastIonData(
            source='icrh',
            species='He3',
            n=jnp.full(geo.rho.shape, 2.5e19),
            n_right_bc=np.float64(0.0),
            T=jnp.full(geo.rho.shape, 75.0),
            T_right_bc=np.float64(0.0),
        ),
    )
    runtime_params_with_pfi = dataclasses.replace(
        runtime_params,
        profile_conditions=dataclasses.replace(
            runtime_params.profile_conditions,
            prescribed_fast_ions=pfi,
        ),
    )

    result, _ = source_profile_builders.build_basic_sources(
        runtime_params=runtime_params_with_pfi,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=is_explicit,
    )
    all_outputs = {**(explicit_outputs or {}), **result}
    self.assertIn('icrh', all_outputs)
    self.assertIn('ecrh', all_outputs)
    self.assertNotIn('ohmic', all_outputs)
    self.assertNotIn('impurity_radiation', all_outputs)

  def test_explicit_source_profiles_not_mutated(self):
    (
        source_models,
        runtime_params,
        geo,
        core_profiles,
        source_states,
        _,
        bootstrap_current,
    ) = self._build_test_env(is_explicit=False)

    explicit_outputs, _ = source_profile_builders.build_basic_sources(
        runtime_params,
        geo,
        core_profiles,
        source_models,
        source_states=source_states,
        explicit=True,
    )

    orig_keys = set(explicit_outputs.keys())

    basic_outputs, stage_states = source_profile_builders.build_basic_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=False,
    )
    all_basic_outputs = {**explicit_outputs, **basic_outputs}
    _ = source_profile_builders.build_remaining_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=stage_states,
        basic_source_outputs=all_basic_outputs,
        bootstrap_current=bootstrap_current,
    )

    self.assertEqual(set(explicit_outputs.keys()), orig_keys)

  def test_build_psi_sources(self):
    (
        source_models,
        runtime_params,
        geo,
        core_profiles,
        source_states,
        _,
        _,
    ) = self._build_test_env(is_explicit=True)

    psi_profiles, _ = source_profile_builders.build_psi_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
    )
    self.assertIn('ecrh', psi_profiles)
    self.assertNotIn('ohmic', psi_profiles)
    self.assertNotIn('icrh', psi_profiles)


if __name__ == '__main__':
  absltest.main()
