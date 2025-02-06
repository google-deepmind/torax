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
from absl.testing import absltest
import chex
from torax import core_profile_setters
from torax import math_utils
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.sources import generic_ion_el_heat_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import (
    impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib,
)
from torax.sources.tests import test_lib


class ImpurityRadiationConstantFractionTest(test_lib.SourceTestCase):
  """Tests impurity_radiation_constant_fraction implementation of ImpurityRadiationHeatSink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink,
        runtime_params_class=impurity_radiation_constant_fraction.RuntimeParams,
        source_name=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME,
        links_back=True,
        model_func=impurity_radiation_constant_fraction.radially_constant_fraction_of_Pin,
    )

  def test_source_value(self):
    """Tests that the source value is correct."""
    # Source builder for this class
    impurity_radiation_sink_builder = self._source_class_builder()
    impurity_radiation_sink_builder.runtime_params.mode = (
        runtime_params_lib.Mode.MODEL_BASED
    )
    if not source_lib.is_source_builder(impurity_radiation_sink_builder):
      raise TypeError(f'{type(self)} has a bad _source_class_builder')

    # Source builder for generic_ion_el_heat_source
    # We don't test this class, as that should be done in its own test
    heat_source_builder_builder = source_lib.make_source_builder(
        source_type=generic_ion_el_heat_source.GenericIonElectronHeatSource,
        runtime_params_type=generic_ion_el_heat_source.RuntimeParams,
        model_func=generic_ion_el_heat_source.default_formula,
    )
    heat_source_builder = heat_source_builder_builder(
        model_func=generic_ion_el_heat_source.default_formula
    )

    # Runtime params
    runtime_params = general_runtime_params.GeneralRuntimeParams()

    # Source models
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {
            self._source_name: impurity_radiation_sink_builder,
            generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME: (
                heat_source_builder
            ),
        },
    )
    source_models = source_models_builder()

    # Extract the source we're testing and check that it's been built correctly
    impurity_radiation_sink = source_models.sources[self._source_name]
    self.assertIsInstance(impurity_radiation_sink, source_lib.Source)

    # Geometry, profiles, and dynamic runtime params
    geo = circular_geometry.build_circular_geometry()
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
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    impurity_radiation_sink_dynamic_runtime_params_slice = (
        dynamic_runtime_params_slice.sources[self._source_name]
    )

    heat_source_dynamic_runtime_params_slice = (
        dynamic_runtime_params_slice.sources[
            generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME
        ]
    )

    assert isinstance(
        impurity_radiation_sink_dynamic_runtime_params_slice,
        impurity_radiation_constant_fraction.DynamicRuntimeParams,
    )
    assert isinstance(
        heat_source_dynamic_runtime_params_slice,
        generic_ion_el_heat_source.DynamicRuntimeParams,
    )
    impurity_radiation_heat_sink_power_density = (
        impurity_radiation_sink.get_value(
            static_runtime_params_slice=static_slice,
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            geo=geo,
            core_profiles=core_profiles,
            calculated_source_profiles=None,
        )
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


if __name__ == '__main__':
  absltest.main()
