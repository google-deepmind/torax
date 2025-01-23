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
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax import math_utils
from torax.config import plasma_composition
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
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
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

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    geo = circular_geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {self._source_name: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[self._source_name]
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


class ImpurityRadiationMavrinFitTest(test_lib.SourceTestCase):
  """Tests impurity_radiation_mavrin_fit implementation of ImpurityRadiationHeatSink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink,
        runtime_params_class=impurity_radiation_mavrin_fit.RuntimeParams,
        source_name=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME,
        model_func=impurity_radiation_mavrin_fit.impurity_radiation_mavrin_fit,
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
        impurity_radiation_mavrin_fit.DynamicRuntimeParams,
    )
    assert isinstance(
        heat_source_dynamic_runtime_params_slice,
        generic_ion_el_heat_source.DynamicRuntimeParams,
    )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    geo = circular_geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {self._source_name: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[self._source_name]
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

  def test_consistency_of_model_function_with_cooling_curve(self):
    """Tests that the model function is consistent with the cooling curve."""
    # Source builder for this class
    impurity_radiation_sink_builder = self._source_class_builder()
    impurity_radiation_sink_builder.runtime_params.mode = (
        runtime_params_lib.Mode.MODEL_BASED
    )
    # Non-unity multiplier to unit-test the runtime params
    impurity_radiation_sink_builder.runtime_params.radiation_multiplier = 0.5

    # General runtime params
    runtime_params = general_runtime_params.GeneralRuntimeParams()

    # Source models
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {
            self._source_name: impurity_radiation_sink_builder,
        },
    )
    source_models = source_models_builder()

    # Geometry, profiles, and dynamic runtime params
    geo = circular_geometry.build_circular_geometry()

    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    radiation_profile = (
        impurity_radiation_mavrin_fit.impurity_radiation_mavrin_fit(
            static_runtime_params_slice=static_slice,
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            geo=geo,
            source_name=self._source_name,
            core_profiles=core_profiles,
        )
    )

    # pylint: disable=invalid-name
    expected_LZ = impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
        ion_symbols=static_slice.impurity_names,
        ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
        Te=core_profiles.temp_el.value,
    )

    expected_radiation_profile = (
        expected_LZ
        * core_profiles.ne.value
        * core_profiles.nimp.value
        * 0.5
        * dynamic_runtime_params_slice.numerics.nref**2
    )
    np.testing.assert_allclose(
        radiation_profile,
        expected_radiation_profile,
    )

  # pylint: disable=invalid-name
  @parameterized.product(
      ion_symbol=[
          ('C',),
          ('N',),
          ('O',),
          ('Ne',),
          ('Ar',),
          ('Kr',),
          ('Xe',),
          ('W',),
      ],
      temperature=[0.1, 1.0, [10.0, 20.0], 90.0],
  )
  def test_calculate_total_impurity_radiation_sanity(
      self, ion_symbol, temperature
  ):
    """Test with valid ions and within temperature range."""
    Te = np.array(temperature)
    ion_mixture = plasma_composition.DynamicIonMixture(
        fractions=np.array([1.0]),
        avg_A=2.0,  # unused
    )
    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            ion_mixture,
            Te,
        )
    )
    np.testing.assert_equal(
        LZ_calculated.shape,
        Te.shape,
        err_msg=(
            f'LZ and T shapes unequal for {ion_symbol} at temperature {Te}. LZ'
            f' = {LZ_calculated}, LZ.shape = {LZ_calculated.shape}, Te.shape ='
            f' {Te.shape}.'
        ),
    )
    # Physical sanity checking
    np.testing.assert_array_less(
        0.0,
        LZ_calculated,
        err_msg=(
            f'Unphysical negative LZ for {ion_symbol} at temperature {Te}. '
            f'LZ = {LZ_calculated}.'
        ),
    )

  @parameterized.named_parameters(
      ('Te_low_input', 0.05, 0.1),
      ('Te_high_input', 150.0, 100.0),
  )
  def test_temperature_clipping(self, Te_input, Te_clipped):
    """Test with valid ions and within temperature range."""
    ion_symbol = ('W',)
    ion_mixture = plasma_composition.DynamicIonMixture(
        fractions=np.array([1.0]),
        avg_A=2.0,  # unused
    )
    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            ion_mixture,
            Te_input,
        )
    )
    LZ_expected = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            ion_mixture,
            Te_clipped,
        )
    )

    np.testing.assert_allclose(
        LZ_calculated,
        LZ_expected,
        err_msg=(
            f'Te clipping not working as expected for Te_input={Te_input},'
            f' LZ_calculated = {LZ_calculated}, Z_expected={LZ_expected}'
        ),
    )

  @parameterized.named_parameters(
      (
          'Helium-3',
          {'He3': 1.0},
          [0.1, 2, 10],
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Helium-4',
          {'He4': 1.0},
          [0.1, 2, 10],
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Lithium',
          {'Li': 1.0},
          [0.1, 2, 10],
          [1.37075024e-35, 9.16765402e-36, 1.60346076e-35],
      ),
      (
          'Beryllium',
          {'Be': 1.0},
          [0.1, 2, 10],
          [6.86895406e-35, 1.88578938e-35, 3.04614535e-35],
      ),
      (
          'Carbon',
          {'C': 1.0},
          [0.1, 2, 10],
          [6.74683566e-34, 5.89332177e-35, 7.94786067e-35],
      ),
      (
          'Nitrogen',
          {'N': 1.0},
          [0.1, 2, 10],
          [6.97912189e-34, 9.68950644e-35, 1.15250226e-34],
      ),
      (
          'Oxygen',
          {'O': 1.0},
          [0.1, 2, 10],
          [4.10676301e-34, 1.57469152e-34, 1.58599054e-34],
      ),
      (
          'Neon',
          {'Ne': 1.0},
          [0.1, 2, 10],
          [1.19151664e-33, 3.27468464e-34, 2.82416557e-34],
      ),
      (
          'Argon',
          {'Ar': 1.0},
          [0.1, 2, 10],
          [1.92265224e-32, 4.02388371e-33, 1.53295491e-33],
      ),
      (
          'Krypton',
          {'Kr': 1.0},
          [0.1, 2, 10],
          [6.57654706e-32, 3.23512795e-32, 7.53285680e-33],
      ),
      (
          'Xenon',
          {'Xe': 1.0},
          [0.1, 2, 10],
          [2.89734288e-31, 8.96916315e-32, 2.87740863e-32],
      ),
      (
          'Tungsten',
          {'W': 1.0},
          [0.1, 2, 10],
          [1.66636258e-31, 4.46651033e-31, 1.31222935e-31],
      ),
      (
          'Mixture',
          {
              'He4': 0.2,
              'Li': 0.1,
              'Be': 0.197,
              'C': 0.1,
              'N': 0.1,
              'O': 0.1,
              'Ne': 0.1,
              'Ar': 0.1,
              'Kr': 0.001,
              'Xe': 0.001,
              'W': 0.001,
          },
          [0.1, 2, 10],
          [2.75762225e-33, 1.04050126e-33, 3.93256085e-34],
      ),
  )
  def test_calculate_total_impurity_radiation(
      self,
      species,
      Te,
      expected_LZ,
  ):
    """Test calculate_total_impurity_radiation.

    Args:
      species: A dictionary of ion symbols and their fractions.
      Te: The temperature in KeV.
      expected_LZ: The expected effective cooling curve value.

    expected_LZ references were verified against plots in the Mavrin 2018 paper.
    """
    Te = np.array(Te)
    expected_LZ = np.array(expected_LZ)
    avg_A = 2.0  # arbitrary, not used.
    ion_symbols = tuple(species.keys())
    fractions = np.array(tuple(species.values()))
    ion_mixture = plasma_composition.DynamicIonMixture(
        fractions=fractions,
        avg_A=avg_A,
    )
    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbols,
            ion_mixture,
            Te,
        )
    )

    np.testing.assert_allclose(LZ_calculated, expected_LZ, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
