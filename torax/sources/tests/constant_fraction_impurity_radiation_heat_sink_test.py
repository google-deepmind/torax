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
from unittest import mock

from absl.testing import absltest
import chex
from torax import math_utils
from torax.config import runtime_params_slice
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import generic_ion_el_heat_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_profiles
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import (
    impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib,
)
from torax.sources.tests import test_lib


class ImpurityRadiationConstantFractionTest(
    test_lib.SingleProfileSourceTestCase
):

  def setUp(self):
    super().setUp(
        source_config_class=impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
        source_name=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME,
        needs_source_models=True,
    )

  def test_source_value(self):
    heat_name = (
        generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME
    )
    impurity_name = (
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME
    )

    impurity_radiation_dynamic = (
        impurity_radiation_constant_fraction.DynamicRuntimeParams(
            prescribed_values=mock.ANY,
            fraction_of_total_power_density=0.5,
        )
    )

    heat_dynamic = generic_ion_el_heat_source.DynamicRuntimeParams(
        prescribed_values=mock.ANY,
        rsource=0.0,
        w=0.25,
        Ptot=120e6,
        el_heat_fraction=0.66666,
    )

    static = runtime_params_lib.StaticRuntimeParams(
        mode=runtime_params_lib.Mode.MODEL_BASED.value,
        is_explicit=False,
    )

    dynamic_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            heat_name: heat_dynamic,
            impurity_name: impurity_radiation_dynamic,
        },
    )

    static_slice = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={heat_name: static, impurity_name: static},
    )

    heat_source = generic_ion_el_heat_source.GenericIonElectronHeatSource(
        model_func=generic_ion_el_heat_source.default_formula,
    )

    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    el, ion = heat_source.get_value(
        static_slice,
        dynamic_slice,
        geo,
        mock.ANY,
        None,
    )

    impurity_radiation_sink = impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink(
        model_func=impurity_radiation_constant_fraction.radially_constant_fraction_of_Pin
    )

    impurity_radiation_heat_sink_power_density = (
        impurity_radiation_sink.get_value(
            static_runtime_params_slice=static_slice,
            dynamic_runtime_params_slice=dynamic_slice,
            geo=geo,
            core_profiles=mock.ANY,
            calculated_source_profiles=source_profiles.SourceProfiles(
                j_bootstrap=mock.ANY,
                qei=mock.ANY,
                temp_el={'foo': el},
                temp_ion={'foo_source': ion},
            ),
        )
    )

    self.assertLen(impurity_radiation_heat_sink_power_density, 1)
    impurity_radiation_heat_sink_power_density = (
        impurity_radiation_heat_sink_power_density[0]
    )
    # The value should be equal to fraction * sum of the (TEMP_EL+TEMP_ION)
    # sources, minus P_ei and P_brems.
    # In this case, that is only the generic_ion_el_heat_source.
    impurity_radiation_heat_sink_power = math_utils.volume_integration(
        impurity_radiation_heat_sink_power_density, geo
    )
    chex.assert_trees_all_close(
        impurity_radiation_heat_sink_power,
        heat_dynamic.Ptot
        * -impurity_radiation_dynamic.fraction_of_total_power_density,
        rtol=1e-2,  # TODO(b/382682284): this rtol seems v. high
    )


if __name__ == '__main__':
  absltest.main()
