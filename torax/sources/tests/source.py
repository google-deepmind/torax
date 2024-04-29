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

"""Tests for source_lib.py."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_zero_profile_works_by_default(self):
    """The default source impl should support profiles with all zeros."""
    source = source_lib.Source(
        output_shape_getter=source_lib.get_cell_profile_shape,
        affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(
        profile,
        source_lib.ProfileType.CELL.get_zero_profile(geo),
    )

  def test_unsupported_modes_raise_errors(self):
    """Calling with an unsupported type should raise an error."""
    source = source_lib.Source(
        supported_modes=(
            # Only support formula-based profiles.
            runtime_params_lib.Mode.FORMULA_BASED,
        ),
        output_shape_getter=source_lib.get_cell_profile_shape,
        affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )
    # But set the runtime params of the source to use ZERO as the mode.
    source.runtime_params.mode = runtime_params_lib.Mode.ZERO
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    # But calling requesting ZERO shouldn't work.
    with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
      source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
              'foo'
          ],
          geo=geo,
          core_profiles=core_profiles,
      )

  def test_defaults_output_zeros(self):
    """The default model and formula implementations should output zeros."""
    source = source_lib.Source(
        supported_modes=(
            runtime_params_lib.Mode.MODEL_BASED,
            runtime_params_lib.Mode.FORMULA_BASED,
        ),
        output_shape_getter=source_lib.get_cell_profile_shape,
        affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    with self.subTest('model_based'):
      dynamic_runtime_params_slice = (
          runtime_params_slice.build_dynamic_runtime_params_slice(
              runtime_params,
              sources={
                  'foo': (
                      dataclasses.replace(
                          source.runtime_params,
                          mode=runtime_params_lib.Mode.MODEL_BASED,
                      )
                  )
              },
          )
      )
      profile = source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
              'foo'
          ],
          geo=geo,
          core_profiles=core_profiles,
      )
      np.testing.assert_allclose(
          profile,
          source_lib.ProfileType.CELL.get_zero_profile(geo),
      )
    with self.subTest('formula'):
      dynamic_runtime_params_slice = (
          runtime_params_slice.build_dynamic_runtime_params_slice(
              runtime_params,
              sources={
                  'foo': (
                      dataclasses.replace(
                          source.runtime_params,
                          mode=runtime_params_lib.Mode.FORMULA_BASED,
                      )
                  )
              },
          )
      )
      profile = source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
              'foo'
          ],
          geo=geo,
          core_profiles=core_profiles,
      )
      np.testing.assert_allclose(
          profile,
          source_lib.ProfileType.CELL.get_zero_profile(geo),
      )

  def test_overriding_default_formula(self):
    """The user-specified formula should override the default formula."""
    output_shape = (2, 4)  # Some arbitrary shape.
    expected_output = jnp.ones(output_shape)
    source = source_lib.Source(
        output_shape_getter=lambda _0: output_shape,
        formula=lambda _0, _1, _2, _3: expected_output,
        affected_core_profiles=(
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.TEMP_EL,
        ),
    )
    source.runtime_params.mode = runtime_params_lib.Mode.FORMULA_BASED
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(profile, expected_output)

  def test_overriding_model(self):
    """The user-specified model should override the default model."""
    output_shape = (2, 4)  # Some arbitrary shape.
    expected_output = jnp.ones(output_shape)
    source = source_lib.Source(
        supported_modes=(runtime_params_lib.Mode.MODEL_BASED,),
        output_shape_getter=lambda _0: output_shape,
        model_func=lambda _0, _1, _2, _3: expected_output,
        affected_core_profiles=(
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.TEMP_EL,
        ),
    )
    source.runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(profile, expected_output)

  def test_retrieving_profile_for_affected_state(self):
    """Grabbing the correct profile works for all mesh state attributes."""
    output_shape = (2, 4)  # Some arbitrary shape.
    profile = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])  # from get_value()
    source = source_lib.Source(
        supported_modes=(runtime_params_lib.Mode.MODEL_BASED,),
        output_shape_getter=lambda _0: output_shape,
        model_func=lambda _0, _1, _2, _3: profile,
        affected_core_profiles=(
            source_lib.AffectedCoreProfile.PSI,
            source_lib.AffectedCoreProfile.NE,
        ),
    )
    geo = geometry.build_circular_geometry(nr=4)
    psi_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.PSI.value, geo
    )
    np.testing.assert_allclose(psi_profile, [1, 2, 3, 4])
    ne_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.NE.value, geo
    )
    np.testing.assert_allclose(ne_profile, [5, 6, 7, 8])
    temp_ion_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_ION.value, geo
    )
    np.testing.assert_allclose(temp_ion_profile, [0, 0, 0, 0])
    temp_el_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_EL.value, geo
    )
    np.testing.assert_allclose(temp_el_profile, [0, 0, 0, 0])


class SingleProfileSourceTest(parameterized.TestCase):
  """Tests for SingleProfileSource."""

  def test_custom_formula(self):
    """The user-specified formula should override the default formula."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry(nr=5)
    expected_output = jnp.ones(5)  # 5 matches the geo.
    source = source_lib.SingleProfileSource(
        formula=lambda _0, _1, _2, _3: expected_output,
        affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )
    source.runtime_params.mode = runtime_params_lib.Mode.FORMULA_BASED
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(profile, expected_output)

  def test_multiple_profiles_raises_error(self):
    """A formula which outputs the wrong shape will raise an error."""
    source = source_lib.SingleProfileSource(
        formula=lambda _0, _1, _2, _3: jnp.ones((2, 5)),
        affected_core_profiles=(
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.NE,
        ),
    )
    source.runtime_params.mode = runtime_params_lib.Mode.FORMULA_BASED
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry(nr=5)
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        # defaults are enough for this.
        source_models=source_models,
    )
    with self.assertRaises(AssertionError):
      source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
              'foo'
          ],
          geo=geo,
          core_profiles=core_profiles,
      )

  def test_retrieving_profile_for_affected_state(self):
    """Grabbing the correct profile works for all mesh state attributes."""
    profile = jnp.asarray([1, 2, 3, 4])  # from get_value()
    source = source_lib.SingleProfileSource(
        supported_modes=(runtime_params_lib.Mode.MODEL_BASED,),
        model_func=lambda _0, _1, _2, _3: profile,
        affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )
    geo = geometry.build_circular_geometry(nr=4)
    psi_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.PSI.value, geo
    )
    np.testing.assert_allclose(psi_profile, [0, 0, 0, 0])
    ne_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.NE.value, geo
    )
    np.testing.assert_allclose(ne_profile, [1, 2, 3, 4])


if __name__ == '__main__':
  absltest.main()
