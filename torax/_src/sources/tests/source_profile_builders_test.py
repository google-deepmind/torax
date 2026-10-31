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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.core_profiles import (
    runtime_params as core_profile_runtime_params,
)
from torax._src.geometry import circular_geometry
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.sources import runtime_params as source_runtime_params
from torax._src.sources import source
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib
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
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.time_step_calculator.t_initial,
    )
    geo = torax_config.geometry.build_provider(
        torax_config.time_step_calculator.t_initial
    )
    source_states = source_models.initial_state(runtime_params)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_model=neoclassical_model,
        source_states=source_states,
    )
    conductivity, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, core_profiles
        )
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        conductivity=conductivity,
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

  def test_computing_standard_source_profiles_for_single_affected_core_profile(
      self,
  ):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class TestSource(source.Source):

      SOURCE_ID = 'foo'
      AFFECTED_CORE_PROFILES = (source.AffectedCoreProfile.PSI,)

      def _get_model_value(self, *args, **kwargs):
        source_state = kwargs.get('source_state', args[-1])
        return (
            source_profiles.SourceOutput(
                profiles=(jnp.ones(self_test.geo.rho.shape),)
            ),
            source_state,
        )

    self_test = self
    test_source = TestSource()
    source_models = mock.create_autospec(
        source_models_lib.SourceModels,
        standard_sources={'foo': test_source},
        psi_sources={},
    )
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            'foo': source_runtime_params.RuntimeParams(
                prescribed_values=(jnp.ones(self.geo.rho.shape),),
                mode=source_runtime_params.Mode.MODEL_BASED,
                is_explicit=True,
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        source_states={'foo': source_state_lib.SourceState()},
        explicit=True,
        calculated_source_profiles=profiles,
    )
    psi_profiles = profiles.psi
    self.assertLen(psi_profiles, 1)
    self.assertIn('foo', psi_profiles)
    np.testing.assert_equal(psi_profiles['foo'].shape, self.geo.rho.shape)

  def test_computing_standard_source_profiles_for_multiple_affected_core_profile(
      self,
  ):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class TestSource(source.Source):

      SOURCE_ID = 'foo'
      AFFECTED_CORE_PROFILES = (
          source.AffectedCoreProfile.TEMP_ION,
          source.AffectedCoreProfile.TEMP_EL,
      )

      def _get_model_value(self, *args, **kwargs):
        source_state = kwargs.get('source_state', args[-1])
        return (
            source_profiles.SourceOutput(
                profiles=(jnp.ones_like(self_test.geo.rho),) * 2
            ),
            source_state,
        )

    self_test = self
    test_source = TestSource()
    source_models = mock.create_autospec(
        source_models_lib.SourceModels,
        standard_sources={'foo': test_source},
        psi_sources={},
    )
    dynamic_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            'foo': source_runtime_params.RuntimeParams(
                prescribed_values=(
                    jnp.ones(self.geo.rho.shape),
                    jnp.ones(self.geo.rho.shape),
                ),
                mode=source_runtime_params.Mode.MODEL_BASED,
                is_explicit=True,
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        runtime_params=dynamic_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        source_states={'foo': source_state_lib.SourceState()},
        explicit=True,
        calculated_source_profiles=profiles,
    )

    # Check that a single profile is returned for each affected core profile.
    # These profiles should be the same shape as the geo.rho.
    ion_profiles = profiles.T_i
    self.assertLen(ion_profiles, 1)
    self.assertIn('foo', ion_profiles)
    np.testing.assert_equal(ion_profiles['foo'].shape, self.geo.rho.shape)

    el_profiles = profiles.T_e
    self.assertLen(el_profiles, 1)
    self.assertIn('foo', el_profiles)
    np.testing.assert_equal(el_profiles['foo'].shape, self.geo.rho.shape)

  @parameterized.parameters(
      dict(
          calculate_anyway=True,
          is_explicit=True,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=True,
          is_explicit=False,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=False,
          is_explicit=True,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=False,
          is_explicit=False,
          expected_calculate=False,
      ),
  )
  def test_build_standard_source_profiles_calculate_anyway(
      self, calculate_anyway, is_explicit, expected_calculate
  ):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class TestSource(source.Source):

      SOURCE_ID = 'foo'
      AFFECTED_CORE_PROFILES = (source.AffectedCoreProfile.PSI,)

      def _get_model_value(self, *args, **kwargs):
        source_state = kwargs.get('source_state', args[-1])
        return (
            source_profiles.SourceOutput(
                profiles=(jnp.ones(self_test.geo.rho.shape),)
            ),
            source_state,
        )

    self_test = self
    test_source = TestSource()
    source_models = mock.create_autospec(
        source_models_lib.SourceModels,
        standard_sources={'foo': test_source},
        psi_sources={},
    )
    dynamic_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            'foo': source_runtime_params.RuntimeParams(
                prescribed_values=(jnp.ones(self.geo.rho.shape),),
                mode=source_runtime_params.Mode.MODEL_BASED,
                is_explicit=True,  # Set the source to be explicit.
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        runtime_params=dynamic_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        source_states={'foo': source_state_lib.SourceState()},
        explicit=is_explicit,
        calculated_source_profiles=profiles,
        calculate_anyway=calculate_anyway,
    )

    if expected_calculate:
      self.assertIn('foo', profiles.psi)
    else:
      self.assertNotIn('foo', profiles.psi)

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
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.time_step_calculator.t_initial,
    )
    geo = torax_config.geometry.build_provider(
        torax_config.time_step_calculator.t_initial
    )
    source_states = source_models.initial_state(runtime_params)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_model=neoclassical_model,
        source_states=source_states,
    )
    conductivity, bootstrap_current, _, _ = (
        neoclassical_model.calculate_conductivity_current_and_rotation(
            runtime_params, geo, core_profiles
        )
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        conductivity=conductivity,
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
        explicit_source_outputs=explicit_outputs,
    )
    self.assertIn('icrh', result)
    self.assertIn('ecrh', result)
    self.assertNotIn('ohmic', result)
    self.assertNotIn('impurity_radiation', result)

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
        explicit_source_outputs=explicit_outputs,
    )
    _ = source_profile_builders.build_remaining_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=stage_states,
        basic_source_outputs=basic_outputs,
        bootstrap_current=bootstrap_current,
    )

    self.assertEqual(set(explicit_outputs.keys()), orig_keys)

  def test_source_builders_raise_value_error_missing_implicit_args(self):
    (
        source_models,
        runtime_params,
        geo,
        core_profiles,
        source_states,
        _,
        _,
    ) = self._build_test_env(is_explicit=False)

    with self.assertRaisesRegex(
        ValueError,
        '`explicit_source_outputs` must be provided if explicit is False.',
    ):
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=False,
          explicit_source_outputs=None,
      )

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
