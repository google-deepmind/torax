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

"""Unit tests for the `torax.config.plasma_composition` module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import interpolated_param
from torax.config import plasma_composition
from torax.geometry import geometry


class PlasmaCompositionTest(parameterized.TestCase):
  """Unit tests for methods in the `torax.config.plasma_composition` module."""

  def test_plasma_composition_make_provider(self):
    """Checks provider construction with no issues."""
    pc = plasma_composition.PlasmaComposition()
    geo = geometry.build_circular_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

  @parameterized.parameters(
      (1.0,),
      (1.6,),
      (2.5,),
  )
  def test_zeff_accepts_float_inputs(self, zeff: float):
    """Tests that zeff accepts a single float input."""
    geo = geometry.build_circular_geometry()
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

    geo = geometry.build_circular_geometry()
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
    geo = geometry.build_circular_geometry()
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
  )
  def test_ion_mixture_constructor(self, input_species, should_raise):
    """Tests various cases of IonMixture construction."""
    if should_raise:
      with self.assertRaises(ValueError):
        plasma_composition.IonMixture(species=input_species)
    else:
      plasma_composition.IonMixture(species=input_species)


if __name__ == '__main__':
  absltest.main()
