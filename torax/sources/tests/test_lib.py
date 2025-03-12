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

from typing import Type

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import base
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  def setUp(
      self,
      source_name: str,
      source_config_class: Type[base.SourceModelBase],
      needs_source_models: bool = False,
  ):
    self._source_name = source_name
    self._source_config_class = source_config_class
    self._needs_source_models = needs_source_models
    super().setUp()

  def test_build_dynamic_params(self):
    source = self._source_config_class.from_dict({})
    self.assertIsInstance(source, self._source_config_class)
    torax_pydantic.set_grid(
        source,
        torax_pydantic.Grid1D.construct(nx=4, dx=0.25,),
    )
    dynamic_params = source.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_params, runtime_params_lib.DynamicRuntimeParams
    )

  @parameterized.product(
      mode=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.MODEL_BASED,
          runtime_params_lib.Mode.PRESCRIBED,
      ),
      is_explicit=(True, False),
  )
  def test_runtime_params_builds_static_params(
      self, mode: runtime_params_lib.Mode, is_explicit: bool
  ):
    """Tests that the static params are built correctly."""
    source_config = self._source_config_class.from_dict(
        {'mode': mode, 'is_explicit': is_explicit}
    )
    static_params = source_config.build_static_params()
    self.assertIsInstance(static_params, runtime_params_lib.StaticRuntimeParams)
    self.assertEqual(static_params.mode, mode.value)
    self.assertEqual(static_params.is_explicit, is_explicit)


class SingleProfileSourceTestCase(SourceTestCase):
  """Base test class for SingleProfileSource subclasses."""

  def test_source_value_on_the_cell_grid(self):
    """Tests that the source can provide a value by default on the cell grid."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    sources = source_pydantic_model.Sources.from_dict({self._source_name: {}})
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    if self._needs_source_models:
      calculated_source_profiles = source_profiles.SourceProfiles(
          j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
          psi={'foo': jnp.full(geo.rho.shape, 13.0)},
          temp_el={'foo_source': jnp.full(geo.rho.shape, 17.0)},
          temp_ion={'foo_sink': jnp.full(geo.rho.shape, 19.0)},
          ne={},
          qei=source_profiles.QeiInfo.zeros(geo),
      )
    else:
      calculated_source_profiles = None
    source = source_models.sources[self._source_name]
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
    )[0]
    chex.assert_rank(value, 1)
    self.assertEqual(value.shape, geo.rho.shape)


class IonElSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_values_on_the_cell_grid(self):
    """Tests that the source can provide values on the cell grid."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    sources = source_pydantic_model.Sources.from_dict({self._source_name: {}})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources[self._source_name]
    self.assertIsInstance(source, source_lib.Source)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    ion_and_el = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )
    self.assertLen(ion_and_el, 2)
    self.assertEqual(ion_and_el[0].shape, geo.rho.shape)
    self.assertEqual(ion_and_el[1].shape, geo.rho.shape)
