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

"""Utilities to help with testing sources."""

from collections.abc import Mapping
import dataclasses
from typing import Any, ClassVar, Type

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import pydantic
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic

# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  source_name: ClassVar[str]
  source_config_class: ClassVar[Type[base.SourceConfigBase]]
  needs_source_models: ClassVar[bool] = False
  can_be_explicit: ClassVar[bool] = True
  model_name: ClassVar[str | None] = None
  extra_source_config_kwargs: ClassVar[Mapping[str, Any]] = {}

  def setUp(self):
    super().setUp()
    self._source_name = self.source_name
    self._model_name = self.model_name
    self._source_config_class = self.source_config_class
    self._needs_source_models = self.needs_source_models
    self._can_be_explicit = self.can_be_explicit

  def test_build_runtime_params(self):
    source = self._source_config_class.from_dict(
        dict(self.extra_source_config_kwargs)
    )
    self.assertIsInstance(source, self._source_config_class)
    mesh = torax_pydantic.Grid1D(
        face_centers=interpolated_param_2d.get_face_centers(nx=4)
    )
    torax_pydantic.set_grid(source, mesh)
    runtime_params = source.build_runtime_params(t=0.0)
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)

  @parameterized.product(
      mode=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.MODEL_BASED,
          runtime_params_lib.Mode.PRESCRIBED,
      ),
      is_explicit=(True, False),
  )
  def test_runtime_params_builds_runtime_params(
      self, mode: runtime_params_lib.Mode, is_explicit: bool
  ):
    """Tests that the runtime params builds runtime params."""
    if is_explicit and not self._can_be_explicit:
      with self.assertRaises(pydantic.ValidationError):
        self._source_config_class.from_dict({
            'mode': mode,
            'is_explicit': is_explicit,
            **self.extra_source_config_kwargs,
        })
      return
    source_config = self._source_config_class.from_dict({
        'mode': mode,
        'is_explicit': is_explicit,
        **self.extra_source_config_kwargs,
    })
    mesh = torax_pydantic.Grid1D(
        face_centers=interpolated_param_2d.get_face_centers(nx=4)
    )
    torax_pydantic.set_grid(source_config, mesh)
    runtime_params = source_config.build_runtime_params(t=0.0)
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)
    self.assertEqual(runtime_params.mode, mode)
    self.assertEqual(runtime_params.is_explicit, is_explicit)


class SingleProfileSourceTestCase(SourceTestCase):
  """Base test class for SingleProfileSource subclasses."""

  def test_source_value_on_the_cell_grid(self):
    """Tests that the source can provide a value by default on the cell grid."""
    config_dict = default_configs.get_default_config_dict()
    if self._model_name is not None:
      config_dict['sources'] = {
          self._source_name: {
              'model_name': self._model_name,
          }
      }
    else:
      config_dict['sources'] = {self._source_name: {}}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    if self._needs_source_models:
      calculated_source_profiles = source_profiles.SourceProfiles(
          bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
          psi={'foo': jnp.full(geo.rho.shape, 13.0)},
          T_e={'foo_source': jnp.full(geo.rho.shape, 17.0)},
          T_i={'foo_sink': jnp.full(geo.rho.shape, 19.0)},
          n_e={},
          qei=source_profiles.QeiInfo.zeros(geo),
      )
      conductivity, _, _, _ = (
          neoclassical_model.calculate_conductivity_current_and_rotation(
              runtime_params,
              geo,
              core_profiles,
              fast_ions=core_profiles.fast_ions,
          )
      )
      core_profiles = dataclasses.replace(
          core_profiles,
          conductivity=conductivity,
      )
    else:
      calculated_source_profiles = None
    source = source_models.standard_sources[self._source_name]
    if isinstance(source, source_lib.FinalSource):
      assert calculated_source_profiles is not None
      output, _ = source.get_value(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          calculated_source_profiles=calculated_source_profiles,
          source_state=source.initial_state(runtime_params),
      )
    elif isinstance(
        source, (source_lib.Source, source_lib.SourceUsingCoreProfiles)
    ):
      output, _ = source.get_value(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_state=source.initial_state(runtime_params),
      )
    else:
      raise TypeError(f'Unsupported source type: {type(source)}')
    value = output.profiles[0]
    chex.assert_rank(value, 1)
    self.assertEqual(value.shape, geo.rho.shape)


class MultipleProfileSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_values_on_the_cell_grid(self):
    """Tests that the source can provide values on the cell grid."""
    config_dict = default_configs.get_default_config_dict()
    if self._model_name is not None:
      config_dict['sources'] = {
          self._source_name: {
              'model_name': self._model_name,
          }
      }
    else:
      config_dict['sources'] = {self._source_name: {}}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[self._source_name]
    self.assertIsInstance(source, source_lib.Source)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    value = output.profiles
    self.assertLen(value, 2)
    self.assertEqual(value[0].shape, geo.rho.shape)
    self.assertEqual(value[1].shape, geo.rho.shape)

    self.assertFalse(jnp.any(jnp.isnan(value[0])))
    self.assertFalse(jnp.any(jnp.isnan(value[1])))
