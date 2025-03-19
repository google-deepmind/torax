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
from torax.sources import generic_ion_el_heat_source
from torax.sources.tests import test_lib
import numpy as np
import jax.numpy as jnp
import jax
from torax.core_profiles import initialization
from torax.geometry.pydantic_model import CircularConfig
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax import math_utils
from torax.sources import pydantic_model as source_pydantic_model


def create_sources_pydantic(source_dict):
  """Creates a Sources object from a dictionary."""
  return source_pydantic_model.Sources.from_dict(source_dict)


class GenericIonElectronHeatSourceTest(test_lib.IonElSourceTestCase):
  """Tests for GenericIonElHeatSource."""

  def setUp(self):
    super().setUp(
        source_config_class=generic_ion_el_heat_source.GenericIonElHeatSourceConfig,
        source_name=generic_ion_el_heat_source.GenericIonElectronHeatSource.SOURCE_NAME,
    )

  def _source_class_builder(self):
    """Returns a function that builds a source instance."""
    return lambda: generic_ion_el_heat_source.GenericIonElHeatSourceConfig()

  def test_absorption_fraction(self):
    """Tests that absorption_fraction correctly affects power calculations."""
    # Create test geometry
    geo = CircularConfig().build_geometry()

    # Create runtime params and source models
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_config = generic_ion_el_heat_source.GenericIonElHeatSourceConfig()
    source_model_config = {self._source_name: source_config}
    
    # Create sources dictionary
    sources = {
        self._source_name: source_config.build_source(),
    }
    
    source_models = source_models_lib.SourceModels(
        sources=source_model_config
    )
    
    # Create dynamic and static runtime params slices
    source_pydantic = create_sources_pydantic({self._source_name: {}})
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_pydantic,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=source_pydantic,
        torax_mesh=geo.torax_mesh,
    )

    # Initialize core profiles
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    # Create source instance
    source_instance = generic_ion_el_heat_source.GenericIonElHeatSource(
        model_func=generic_ion_el_heat_source.model_func
    )

    # Create runtime params with different absorption fractions
    rp1 = generic_ion_el_heat_source.DynamicRuntimeParams(
        Ptot=1.0,
        rsource=0.5,
        w=0.2,
        el_heat_fraction=0.5,
        absorption_fraction=1.0,
    )

    rp2 = generic_ion_el_heat_source.DynamicRuntimeParams(
        Ptot=1.0,
        rsource=0.5,
        w=0.2,
        el_heat_fraction=0.5,
        absorption_fraction=0.5,
    )

    # Create dynamic runtime params slices with our test parameters
    dynamic_slice1 = runtime_params_slice.DynamicRuntimeParamsSlice(
        sources={
            'generic_ion_el_heat_source': rp1
        },
        numerics=dynamic_runtime_params_slice.numerics,
        plasma_composition=dynamic_runtime_params_slice.plasma_composition,
        transport=dynamic_runtime_params_slice.transport,
        stepper=dynamic_runtime_params_slice.stepper,
        profile_conditions=dynamic_runtime_params_slice.profile_conditions,
        pedestal=dynamic_runtime_params_slice.pedestal,
    )

    dynamic_slice2 = runtime_params_slice.DynamicRuntimeParamsSlice(
        sources={
            'generic_ion_el_heat_source': rp2
        },
        numerics=dynamic_runtime_params_slice.numerics,
        plasma_composition=dynamic_runtime_params_slice.plasma_composition,
        transport=dynamic_runtime_params_slice.transport,
        stepper=dynamic_runtime_params_slice.stepper,
        profile_conditions=dynamic_runtime_params_slice.profile_conditions,
        pedestal=dynamic_runtime_params_slice.pedestal,
    )

    # Get profiles using get_value
    ion1, el1 = source_instance.get_value(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_slice1,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )

    ion2, el2 = source_instance.get_value(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_slice2,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )

    # Test that the absorbed power is scaled by absorption_fraction
    # Profile 2 should have half the power of profile 1
    integrated_ion1 = math_utils.volume_integration(ion1, geo)
    integrated_ion2 = math_utils.volume_integration(ion2, geo)
    integrated_el1 = math_utils.volume_integration(el1, geo)
    integrated_el2 = math_utils.volume_integration(el2, geo)

    np.testing.assert_allclose(integrated_ion2 / integrated_ion1, 0.5, rtol=1e-5)
    np.testing.assert_allclose(integrated_el2 / integrated_el1, 0.5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
