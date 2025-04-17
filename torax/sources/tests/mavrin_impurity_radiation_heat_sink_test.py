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
from absl.testing import parameterized
import numpy as np
from torax.config import plasma_composition
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax.sources.tests import test_lib
from torax.torax_pydantic import torax_pydantic


class MarvinImpurityRadiationHeatSinkTest(test_lib.SingleProfileSourceTestCase):

  def setUp(self):
    super().setUp(
        source_config_class=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
        source_name=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME,
    )

  def test_correct_dynamic_params_built(self):
    # Source models
    sources = sources_pydantic_model.Sources.from_dict({
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME: {
        },
    })
    # Set the grid to allows the dynamic params to be built without making the
    # full config.
    torax_pydantic.set_grid(sources, torax_pydantic.Grid1D(nx=4, dx=0.25))
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    runtime_params = sources.source_model_config[
        self._source_name
    ].build_dynamic_params(t=0.0)

    # Extract the source we're testing and check that it's been built correctly
    impurity_radiation_sink = source_models.sources[self._source_name]
    self.assertIsInstance(impurity_radiation_sink, source_lib.Source)

    assert isinstance(
        runtime_params,
        impurity_radiation_mavrin_fit.DynamicRuntimeParams,
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
