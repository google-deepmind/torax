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
from torax import interpolated_param
from torax.config import plasma_composition
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.physics import charge_states


class PlasmaCompositionTest(parameterized.TestCase):

  def test_plasma_composition_make_provider(self):
    """Checks provider construction with no issues."""
    pc = plasma_composition.PlasmaComposition()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

  @parameterized.parameters(
      (1.0,),
      (1.6,),
      (2.5,),
  )
  def test_zeff_accepts_float_inputs(self, zeff: float):
    """Tests that zeff accepts a single float input."""
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    pc = plasma_composition.PlasmaComposition(Zeff=zeff)
    provider = pc.make_provider(geo.torax_mesh)
    dynamic_pc = provider.build_dynamic_params(t=0.0)
    # Check that the values in both Zeff and Zeff_face are the same
    # and consistent with the zeff float input
    np.testing.assert_allclose(
        dynamic_pc.Zeff,
        zeff,
    )
    np.testing.assert_allclose(
        dynamic_pc.Zeff_face,
        zeff,
    )

  def test_zeff_and_zeff_face_match_expected(self):
    """Checks that Zeff and Zeff_face are calculated as expected."""
    # Define an arbitrary Zeff profile
    zeff_profile = {
        0.0: {0.0: 1.0, 0.5: 1.3, 1.0: 1.6},
        1.0: {0.0: 1.8, 0.5: 2.1, 1.0: 2.4},
    }

    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    pc = plasma_composition.PlasmaComposition(Zeff=zeff_profile)
    provider = pc.make_provider(geo.torax_mesh)

    # Check values at t=0.0
    dynamic_pc = provider.build_dynamic_params(t=0.0)
    expected_zeff = np.interp(
        geo.rho_norm,
        np.array(list(zeff_profile[0.0])),
        np.array(list(zeff_profile[0.0].values())),
    )
    expected_zeff_face = np.interp(
        geo.rho_face_norm,
        np.array(list(zeff_profile[0.0])),
        np.array(list(zeff_profile[0.0].values())),
    )
    np.testing.assert_allclose(dynamic_pc.Zeff, expected_zeff)
    np.testing.assert_allclose(dynamic_pc.Zeff_face, expected_zeff_face)

    # Check values at t=0.5 (interpolated in time)
    dynamic_pc = provider.build_dynamic_params(t=0.5)
    expected_zeff = np.interp(
        geo.rho_norm,
        np.array([0.0, 0.5, 1.0]),
        np.array([1.4, 1.7, 2.0]),
    )
    expected_zeff_face = np.interp(
        geo.rho_face_norm,
        np.array([0.0, 0.5, 1.0]),
        np.array([1.4, 1.7, 2.0]),
    )
    np.testing.assert_allclose(dynamic_pc.Zeff, expected_zeff)
    np.testing.assert_allclose(dynamic_pc.Zeff_face, expected_zeff_face)

  def test_interpolated_vars_are_only_constructed_once(
      self,
  ):
    """Tests that interpolated vars are only constructed once."""
    pc = plasma_composition.PlasmaComposition()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    interpolated_params = {}
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        interpolated_params[field] = value

    # Check we don't make any additional calls to construct interpolated vars.
    provider.build_dynamic_params(t=1.0)
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        self.assertIs(value, interpolated_params[field])

  def test_get_ion_names(self):
    # Test the get_ion_names method
    pc = plasma_composition.PlasmaComposition(
        main_ion={'D': 0.5, 'T': 0.5}, impurity='Ar'
    )
    main_ion_names = pc.get_main_ion_names()
    impurity_names = pc.get_impurity_names()
    self.assertEqual(main_ion_names, ('D', 'T'))
    self.assertEqual(impurity_names, ('Ar',))


class IonMixtureTest(parameterized.TestCase):
  """Unit tests for constructing the IonMixture class."""

  @parameterized.named_parameters(
      ('valid_constant', {'D': 0.5, 'T': 0.5}, False),
      (
          'valid_time_dependent',
          {'D': {0: 0.6, 1: 0.7}, 'T': {0: 0.4, 1: 0.3}},
          False,
      ),
      (
          'valid_time_dependent_with_step',
          {'D': ({0: 0.6, 1: 0.7}, 'step'), 'T': {0: 0.4, 1: 0.3}},
          False,
      ),
      ('invalid_empty', {}, True),
      ('invalid_fractions', {'D': 0.4, 'T': 0.3}, True),
      (
          'invalid_time_mismatch',
          {'D': {0: 0.5, 1: 0.6}, 'T': {0: 0.5, 2: 0.4}},
          True,
      ),
      (
          'invalid_time_dependent_fractions',
          {'D': {0: 0.6, 1: 0.7}, 'T': {0: 0.5, 1: 0.4}},
          True,
      ),
      ('valid_tolerance', {'D': 0.49999999, 'T': 0.5}, False),
      ('invalid_tolerance', {'D': 0.4999, 'T': 0.5}, True),
      ('invalid_not_mapping', 'D', True),
      ('invalid_ion_symbol', {'De': 0.5, 'Tr': 0.5}, True),
  )
  def test_ion_mixture_constructor(self, input_species, should_raise):
    """Tests various cases of IonMixture construction."""
    if should_raise:
      with self.assertRaises(ValueError):
        plasma_composition.IonMixture(species=input_species)
    else:
      plasma_composition.IonMixture(species=input_species)

  # pylint: disable=invalid-name
  @parameterized.named_parameters(
      ('D_constant', {'D': 1.0}, 0.0, 1.0, 2.0141),
      ('T_constant', {'T': 1.0}, 0.0, 1.0, 3.016),
      ('DT_constant_mix_t0', {'D': 0.5, 'T': 0.5}, 0.0, 1.0, 2.51505),
      ('DT_constant_mix_t1', {'D': 0.5, 'T': 0.5}, 1.0, 1.0, 2.51505),
      (
          'DT_time_dependent_mix_t075',
          {'D': {0: 0.0, 1: 1.0}, 'T': {0: 1.0, 1: 0.0}},
          0.75,
          1.0,
          2.264575,
      ),
      (
          'NeC_time_dependent_mix_t075',
          {'C': {0: 0.0, 1: 1.0}, 'Ne': {0: 1.0, 1: 0.0}},
          0.75,
          7.0,
          14.05325,
      ),
  )
  def test_ion_mixture_averaging(self, species, time, expected_Z, expected_A):
    """Tests the averaging of Z and A for different mixtures."""

    mixture = plasma_composition.IonMixture(species=species)
    provider = mixture.make_provider()
    dynamic_mixture = provider.build_dynamic_params(time)
    calculated_Z = charge_states.get_average_charge_state(
        ion_symbols=tuple(species.keys()),
        ion_mixture=dynamic_mixture,
        Te=np.array(10.0),  # Ensure that all ions in test are fully ionized
    )
    np.testing.assert_allclose(calculated_Z, expected_Z)
    np.testing.assert_allclose(dynamic_mixture.avg_A, expected_A)

  @parameterized.named_parameters(
      ('no_override', None, None, 1.0, 2.0141),
      ('Z_override', 3.0, None, 1.0, 2.0141),
      ('A_override', None, 3.0, 1.0, 2.0141),
      ('both_override', 3.0, 3.0, 1.0, 2.0141),
  )
  def test_ion_mixture_override(self, Z_override, A_override, Z, A):
    """Tests overriding the automatic Z/A averaging."""

    mixture = plasma_composition.IonMixture(
        species={'D': {0: 1.0}},
        Z_override=Z_override,
        A_override=A_override,
    )
    provider = mixture.make_provider()
    dynamic_mixture = provider.build_dynamic_params(t=0.0)
    calculated_Z = charge_states.get_average_charge_state(
        ion_symbols=tuple(mixture.species.keys()),
        ion_mixture=dynamic_mixture,
        Te=np.array(1.0),  # arbitrary temperature, won't be used for D
    )
    Z_expected = Z if Z_override is None else Z_override
    A_expected = A if A_override is None else A_override
    np.testing.assert_allclose(calculated_Z, Z_expected)
    np.testing.assert_allclose(dynamic_mixture.avg_A, A_expected)

  def test_from_config(self):
    """Test that IonMixture.from_config behaves as expected."""
    # Single ion.
    mixture = plasma_composition.IonMixture.from_config('D')
    self.assertEqual(mixture.species, {'D': 1.0})

    # Multiple ions.
    mixture = plasma_composition.IonMixture.from_config({'D': 0.6, 'T': 0.4})
    self.assertEqual(set(mixture.species.keys()), {'D', 'T'})

    # Check overrides.
    mixture = plasma_composition.IonMixture.from_config(
        'D', Z_override=1.2, A_override=2.4
    )
    self.assertEqual(mixture.Z_override, 1.2)
    self.assertEqual(mixture.A_override, 2.4)

  # pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
