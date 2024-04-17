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

"""Unit tests for torax.config_slice."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import config as config_lib
from torax import config_slice as config_slice_lib
from torax.sources import formula_config
from torax.sources import source_config
from torax.sources import source_models as source_models_lib


class ConfigSliceTest(parameterized.TestCase):
  """Unit tests for the `config_slice` module."""

  def test_dynamic_slice_can_be_input_to_jitted_function(self):
    def foo(config_slice: config_slice_lib.DynamicConfigSlice):
      _ = config_slice  # do nothing.

    foo_jitted = jax.jit(foo)
    config = config_lib.Config()
    dynamic_slice = config_slice_lib.build_dynamic_config_slice(config)
    # Make sure you can call the function with dynamic_slice as an arg.
    foo_jitted(dynamic_slice)

  def test_dynamic_sources_config_contains_all_sources(self):
    """Tests that all the Sources attributes are covered by SourcesConfig."""
    source_models = source_models_lib.SourceModels().all_sources.keys()
    sources_config_fields = config_slice_lib.build_dynamic_config_slice(
        config_lib.Config()
    ).sources.keys()
    self.assertSameElements(source_models, sources_config_fields)

  def test_time_dependent_provider_is_time_dependent(self):
    """Tests that the config slice provider is time dependent."""
    config = config_lib.Config(
        profile_conditions=config_lib.ProfileConditions(
            Ti_bound_right={0.0: 2.0, 4.0: 4.0},
        ),
    )
    provider = config_slice_lib.TimeDependentDynamicConfigSliceProvider(config)
    dynamic_config_slice = provider(t=1.0)
    np.testing.assert_allclose(
        dynamic_config_slice.profile_conditions.Ti_bound_right, 2.5
    )
    dynamic_config_slice = provider(t=2.0)
    np.testing.assert_allclose(
        dynamic_config_slice.profile_conditions.Ti_bound_right, 3.0
    )

  def test_boundary_conditions_are_time_dependent(self):
    """Tests that the boundary conditions are time dependent params."""
    # All of the following parameters are time-dependent fields, but they can
    # be initialized in different ways.
    config = config_lib.Config(
        profile_conditions=config_lib.ProfileConditions(
            Ti_bound_right={0.0: 2.0, 4.0: 4.0},
            Te_bound_right=4.5,  # not time-dependent.
            ne_bound_right=config_lib.InterpolationParam(
                {5.0: 6.0, 7.0: 8.0},
                interpolation_mode=config_lib.InterpolationMode.STEP,
            ),
        ),
    )
    np.testing.assert_allclose(
        config_slice_lib.build_dynamic_config_slice(
            config, 2.0
        ).profile_conditions.Ti_bound_right,
        3.0,
    )
    np.testing.assert_allclose(
        config_slice_lib.build_dynamic_config_slice(
            config, 4.0
        ).profile_conditions.Te_bound_right,
        4.5,
    )
    np.testing.assert_allclose(
        config_slice_lib.build_dynamic_config_slice(
            config, 6.0
        ).profile_conditions.ne_bound_right,
        6.0,
    )

  def test_pedestal_is_time_dependent(self):
    """Tests that the pedestal config params are time dependent."""
    config = config_lib.Config(
        profile_conditions=config_lib.ProfileConditions(
            set_pedestal={0.0: True, 1.0: False},
            Tiped={0.0: 0.0, 1.0: 1.0},
            Teped={0.0: 1.0, 1.0: 2.0},
            neped={0.0: 2.0, 1.0: 3.0},
            Ped_top={0.0: 3.0, 1.0: 5.0},
        ),
    )
    # Check at time 0.
    dcs = config_slice_lib.build_dynamic_config_slice(config, 0.0)
    profile_conditions = dcs.profile_conditions
    np.testing.assert_allclose(profile_conditions.set_pedestal, True)
    np.testing.assert_allclose(profile_conditions.Tiped, 0.0)
    np.testing.assert_allclose(profile_conditions.Teped, 1.0)
    np.testing.assert_allclose(profile_conditions.neped, 2.0)
    np.testing.assert_allclose(profile_conditions.Ped_top, 3.0)
    # And check after the time limit.
    dcs = config_slice_lib.build_dynamic_config_slice(config, 1.0)
    profile_conditions = dcs.profile_conditions
    np.testing.assert_allclose(profile_conditions.set_pedestal, False)
    np.testing.assert_allclose(profile_conditions.Tiped, 1.0)
    np.testing.assert_allclose(profile_conditions.Teped, 2.0)
    np.testing.assert_allclose(profile_conditions.neped, 3.0)
    np.testing.assert_allclose(profile_conditions.Ped_top, 5.0)

  def test_source_formula_config_has_time_dependent_params(self):
    """Tests that the source formula config params are time-dependent."""
    with self.subTest('default_ne_sources'):
      # Check that the config params for the default ne sources are
      # time-dependent.
      config = config_lib.Config(
          pellet_width={0.0: 0.0, 1.0: 1.0},
          pellet_deposition_location={0.0: 0.0, 1.0: 2.0},
          S_pellet_tot={0.0: 0.0, 1.0: 3.0},
          puff_decay_length={0.0: 0.0, 1.0: 4.0},
          S_puff_tot={0.0: 0.0, 1.0: 5.0},
          nbi_particle_width={0.0: 0.0, 1.0: 6.0},
          nbi_deposition_location={0.0: 0.0, 1.0: 7.0},
          S_nbi_tot={0.0: 0.0, 1.0: 8.0},
      )
      dcs = config_slice_lib.build_dynamic_config_slice(config, 0.5)
      np.testing.assert_allclose(dcs.pellet_width, 0.5)
      np.testing.assert_allclose(dcs.pellet_deposition_location, 1.0)
      np.testing.assert_allclose(dcs.S_pellet_tot, 1.5)
      np.testing.assert_allclose(dcs.puff_decay_length, 2.0)
      np.testing.assert_allclose(dcs.S_puff_tot, 2.5)
      np.testing.assert_allclose(dcs.nbi_particle_width, 3.0)
      np.testing.assert_allclose(dcs.nbi_deposition_location, 3.5)
      np.testing.assert_allclose(dcs.S_nbi_tot, 4.0)

    with self.subTest('exponential_formula'):
      config = config_lib.Config(
          sources={
              'gas_puff_source': source_config.SourceConfig(
                  formula=formula_config.FormulaConfig(
                      exponential=formula_config.Exponential(
                          total={0.0: 0.0, 1.0: 1.0},
                          c1={0.0: 0.0, 1.0: 2.0},
                          c2={0.0: 0.0, 1.0: 3.0},
                      )
                  )
              )
          }
      )
      dcs = config_slice_lib.build_dynamic_config_slice(config, 0.25)
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.exponential.total, 0.25
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.exponential.c1, 0.5
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.exponential.c2, 0.75
      )

    with self.subTest('gaussian_formula'):
      config = config_lib.Config(
          sources={
              'gas_puff_source': source_config.SourceConfig(
                  formula=formula_config.FormulaConfig(
                      gaussian=formula_config.Gaussian(
                          total={0.0: 0.0, 1.0: 1.0},
                          c1={0.0: 0.0, 1.0: 2.0},
                          c2={0.0: 0.0, 1.0: 3.0},
                      )
                  )
              )
          }
      )
      dcs = config_slice_lib.build_dynamic_config_slice(config, 0.25)
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.gaussian.total, 0.25
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.gaussian.c1, 0.5
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.gaussian.c2, 0.75
      )

    with self.subTest('custom_formula'):
      config = config_lib.Config(
          sources={
              'gas_puff_source': source_config.SourceConfig(
                  formula=formula_config.FormulaConfig(
                      custom_params={
                          'foo': 1.0,
                          'bar': {0.0: 0.0, 1.0: 1.0},
                          'baz': config_lib.InterpolationParam(
                              {0.0: 0.0, 1.0: 2.0},
                          ),
                      }
                  )
              )
          }
      )
      dcs = config_slice_lib.build_dynamic_config_slice(config, 1.0)
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.custom_params['foo'], 1.0
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.custom_params['bar'], 1.0
      )
      np.testing.assert_allclose(
          dcs.sources['gas_puff_source'].formula.custom_params['baz'], 2.0
      )

  def test_wext_in_dynamic_config_cannot_be_negative(self):
    """Tests that wext cannot be negative."""
    config = config_lib.Config(wext={0.0: 1.0, 1.0: -1.0})
    # While wext is positive, this should be fine.
    dcs = config_slice_lib.build_dynamic_config_slice(config, 0.0)
    np.testing.assert_allclose(dcs.wext, 1.0)
    # Even 0 should be fine.
    dcs = config_slice_lib.build_dynamic_config_slice(config, 0.5)
    np.testing.assert_allclose(dcs.wext, 0.0)
    # But negative values will cause an error.
    with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
      config_slice_lib.build_dynamic_config_slice(config, 1.0)


if __name__ == '__main__':
  absltest.main()
