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


"""Tests for impurity_radiation_heat_sink."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax import geometry
from torax import math_utils
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import generic_ion_el_heat_source
from torax.sources import (
    impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib,
)
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


class ImpurityRadiationHeatSinkTest(test_lib.SourceTestCase):
  """Tests for ImpurityRadiationHeatSink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink,
        runtime_params_class=impurity_radiation_heat_sink_lib.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
            runtime_params_lib.Mode.PRESCRIBED,
        ],
        links_back=True,
    )

  def test_source_value(self):
    """Tests that the source value is correct."""
    # Source builder for this class
    impurity_radiation_sink_builder = self._source_class_builder()
    impurity_radiation_sink_builder.runtime_params.mode = (
        runtime_params_lib.Mode.MODEL_BASED
    )
    if not source_lib.is_source_builder(impurity_radiation_sink_builder):
      raise TypeError(f"{type(self)} has a bad _source_class_builder")

    # Source builder for generic_ion_el_heat_source
    # We don't test this class, as that should be done in its own test
    heat_source_builder_builder = source_lib.make_source_builder(
        source_type=generic_ion_el_heat_source.GenericIonElectronHeatSource,
        runtime_params_type=generic_ion_el_heat_source.RuntimeParams,
    )
    heat_source_builder = heat_source_builder_builder()

    # Runtime params
    runtime_params = general_runtime_params.GeneralRuntimeParams()

    # Source models
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {
            impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME: (
                impurity_radiation_sink_builder
            ),
            generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME: (
                heat_source_builder
            ),
        },
    )
    source_models = source_models_builder()

    # Extract the source we're testing and check that it's been built correctly
    impurity_radiation_sink = source_models.sources[
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME
    ]
    self.assertIsInstance(impurity_radiation_sink, source_lib.Source)

    # Geometry, profiles, and dynamic runtime params
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    impurity_radiation_sink_dynamic_runtime_params_slice = dynamic_runtime_params_slice.sources[
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME
    ]
    impurity_radiation_sink_static_runtime_params_slice = static_slice.sources[
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME
    ]

    heat_source_dynamic_runtime_params_slice = (
        dynamic_runtime_params_slice.sources[
            generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME
        ]
    )

    assert isinstance(
        impurity_radiation_sink_dynamic_runtime_params_slice,
        impurity_radiation_heat_sink_lib.DynamicRuntimeParams,
    )
    assert isinstance(
        heat_source_dynamic_runtime_params_slice,
        generic_ion_el_heat_source.DynamicRuntimeParams,
    )
    impurity_radiation_heat_sink_power_density = impurity_radiation_sink.get_value(
        static_runtime_params_slice=static_slice,
        static_source_runtime_params=impurity_radiation_sink_static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=impurity_radiation_sink_dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )

    # ImpurityRadiationHeatSink provides TEMP_EL only
    chex.assert_rank(impurity_radiation_heat_sink_power_density, 1)

    # The value should be equal to fraction * sum of the (TEMP_EL+TEMP_ION)
    # sources, minus P_ei and P_brems.
    # In this case, that is only the generic_ion_el_heat_source.
    impurity_radiation_heat_sink_power = math_utils.cell_integration(
        impurity_radiation_heat_sink_power_density * geo.vpr, geo
    )
    chex.assert_trees_all_close(
        impurity_radiation_heat_sink_power,
        heat_source_dynamic_runtime_params_slice.Ptot
        * -impurity_radiation_sink_dynamic_runtime_params_slice.fraction_of_total_power_density,
        rtol=1e-2,  # TODO(b/382682284): this rtol seems v. high
    )

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {"foo": source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources["foo"]
    self.assertIsInstance(source, source_lib.Source)
    dynamic_runtime_params_slice_provider = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )
    )
    # This slice is needed to create the core_profiles
    dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
        t=runtime_params.numerics.t_initial,
    )
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    for unsupported_mode in self._unsupported_modes:
      source_builder.runtime_params.mode = unsupported_mode
      # Construct a new slice with the given mode
      dynamic_runtime_params_slice = (
          runtime_params_slice.DynamicRuntimeParamsSliceProvider(
              runtime_params=runtime_params,
              sources=source_models_builder.runtime_params,
              torax_mesh=geo.torax_mesh,
          )(
              t=runtime_params.numerics.t_initial,
          )
      )
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(RuntimeError):
          source.get_value(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
                  "foo"
              ],
              static_runtime_params_slice=static_runtime_params_slice,
              static_source_runtime_params=static_runtime_params_slice.sources[
                  "foo"
              ],
              geo=geo,
              core_profiles=core_profiles,
          )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    geo = geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {"foo": source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources["foo"]
    self.assertIsInstance(source, source_lib.Source)
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    fake_profile = jnp.ones(cell)
    # Check TEMP_EL is modified
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_EL.value,
            geo,
        ),
        jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.NE.value,
            geo,
        ),
        jnp.zeros(cell),
    )


if __name__ == "__main__":
  absltest.main()
