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
import chex
import jax
import numpy as np
import pydantic
from torax._src import jax_utils
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import electron_density_ratios_zeff
from torax._src.core_profiles.plasma_composition import ion_mixture
from torax._src.core_profiles.plasma_composition import plasma_composition
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.physics import charge_states
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class PlasmaCompositionTest(parameterized.TestCase):

  @parameterized.named_parameters(('too_low', 0.0))
  def test_plasma_composition_validation_error_for_unphysical_zeff(
      self, Z_eff: float
  ):
    with self.assertRaises(pydantic.ValidationError):
      plasma_composition.PlasmaComposition(Z_eff=Z_eff)

  def test_plasma_composition_build_runtime_params_smoke_test(self):
    pc = plasma_composition.PlasmaComposition()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(pc, geo.torax_mesh)
    pc.build_runtime_params(t=0.0)

  @parameterized.parameters(
      (1.0,),
      (1.6,),
      (2.5,),
  )
  def test_zeff_accepts_float_input(self, Z_eff: float):
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    pc = plasma_composition.PlasmaComposition(Z_eff=Z_eff)
    torax_pydantic.set_grid(pc, geo.torax_mesh)
    runtime_params = pc.build_runtime_params(t=0.0)
    # Check that the values in both Z_eff and Z_eff_face are the same
    # and consistent with the zeff float input
    np.testing.assert_allclose(
        runtime_params.Z_eff,
        Z_eff,
    )
    np.testing.assert_allclose(
        runtime_params.Z_eff_face,
        Z_eff,
    )

  def test_zeff_and_zeff_face_match_expected(self):
    # Define an arbitrary Z_eff profile
    zeff_profile = {
        0.0: {0.0: 1.0, 0.5: 1.3, 1.0: 1.6},
        1.0: {0.0: 1.8, 0.5: 2.1, 1.0: 2.4},
    }

    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    pc = plasma_composition.PlasmaComposition(Z_eff=zeff_profile)
    torax_pydantic.set_grid(pc, geo.torax_mesh)

    # Check values at t=0.0
    runtime_params = pc.build_runtime_params(t=0.0)
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
    np.testing.assert_allclose(runtime_params.Z_eff, expected_zeff)
    np.testing.assert_allclose(runtime_params.Z_eff_face, expected_zeff_face)

    # Check values at t=0.5 (interpolated in time)
    runtime_params = pc.build_runtime_params(t=0.5)
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
    np.testing.assert_allclose(runtime_params.Z_eff, expected_zeff)
    np.testing.assert_allclose(runtime_params.Z_eff_face, expected_zeff_face)

  def test_get_ion_names(self):
    # Test the get_ion_names method
    pc = plasma_composition.PlasmaComposition(
        main_ion={'D': 0.5, 'T': 0.5}, impurity='Ar'
    )
    main_ion_names = pc.get_main_ion_names()
    impurity_names = pc.get_impurity_names()
    self.assertEqual(main_ion_names, ('D', 'T'))
    self.assertEqual(impurity_names, ('Ar',))

  @parameterized.named_parameters(
      dict(testcase_name='empty_A_override', A_override=None),
      dict(testcase_name='non_empty_A_override', A_override=1.0),
  )
  def test_plasma_composition_under_jit(self, A_override):
    initial_zeff = 1.5
    updated_zeff = 2.5
    t = 0.0
    pc = plasma_composition.PlasmaComposition(
        Z_eff=initial_zeff, A_i_override=A_override
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(pc, geo.torax_mesh)

    @jax.jit
    def f(pc_model: plasma_composition.PlasmaComposition, t: chex.Numeric):
      return pc_model.build_runtime_params(t=t)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(pc, t)
      # Z_eff is an array, so we check it's all close to the initial value
      chex.assert_trees_all_close(output.Z_eff, initial_zeff)
      if A_override is not None:
        self.assertEqual(output.main_ion.A_avg, A_override)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      pc._update_fields({'Z_eff': updated_zeff})
      # The Z_eff field is a TimeVaryingArray, which gets recreated on update.
      # We need to set the grid again.
      torax_pydantic.set_grid(pc, geo.torax_mesh, mode='relaxed')
      output = f(pc, t)
      chex.assert_trees_all_close(output.Z_eff, updated_zeff)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  @parameterized.named_parameters(
      dict(
          testcase_name='default',
          config={},
          expected_impurity_names=('Ne',),
          expected_Z_override=None,
          expected_A_override=None,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='legacy_impurity_string',
          config={'impurity': 'Ar'},
          expected_impurity_names=('Ar',),
          expected_Z_override=None,
          expected_A_override=None,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='legacy_impurity_dict_single_species',
          config={'impurity': {'Be': 1.0}},
          expected_impurity_names=('Be',),
          expected_Z_override=None,
          expected_A_override=None,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='legacy_impurity_dict_multiple_species',
          config={'impurity': {'Ar': 0.6, 'Ne': 0.4}},
          expected_impurity_names=('Ar', 'Ne'),
          expected_Z_override=None,
          expected_A_override=None,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='legacy_with_overrides',
          config={'impurity': 'Ar', 'Z_impurity_override': 8.0},
          expected_impurity_names=('Ar',),
          expected_Z_override=8.0,
          expected_A_override=None,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='new_api_explicit',
          config={
              'impurity': {
                  'impurity_mode': plasma_composition._IMPURITY_MODE_FRACTIONS,
                  'species': {'C': 0.5, 'N': 0.5},
                  'Z_override': 6.5,
                  'A_override': 13.0,
              },
          },
          expected_impurity_names=('C', 'N'),
          expected_Z_override=6.5,
          expected_A_override=13.0,
          expected_impurity_model_type=ion_mixture.ImpurityFractions,
      ),
      dict(
          testcase_name='new_api_n_e_ratios',
          config={
              'impurity': {
                  'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS,
                  'species': {'C': 0.01, 'N': 0.02},
                  'Z_override': 6.5,
                  'A_override': 13.0,
              },
          },
          expected_impurity_names=('C', 'N'),
          expected_Z_override=6.5,
          expected_A_override=13.0,
          expected_impurity_model_type=electron_density_ratios.ELectronDensityRatios,
      ),
      dict(
          testcase_name='new_api_n_e_ratios_Z_eff',
          config={
              'impurity': {
                  'impurity_mode': (
                      plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF
                  ),
                  'species': {'C': 0.01, 'N': None},
                  'Z_override': 6.5,
                  'A_override': 13.0,
              },
          },
          expected_impurity_names=('C', 'N'),
          expected_Z_override=6.5,
          expected_A_override=13.0,
          expected_impurity_model_type=electron_density_ratios_zeff.ElectronDensityRatiosZeff,
      ),
  )
  def test_impurity_api(
      self,
      config,
      expected_impurity_names,
      expected_Z_override,
      expected_A_override,
      expected_impurity_model_type,
  ):
    pc = plasma_composition.PlasmaComposition(**config)
    self.assertIsInstance(pc.impurity, expected_impurity_model_type)
    self.assertEqual(pc.get_impurity_names(), expected_impurity_names)
    if pc.impurity.Z_override is not None:
      self.assertEqual(
          pc.impurity.Z_override.get_value(0.0), expected_Z_override
      )
    else:
      self.assertIsNone(expected_Z_override)
    if pc.impurity.A_override is not None:
      self.assertEqual(
          pc.impurity.A_override.get_value(0.0), expected_A_override
      )
    else:
      self.assertIsNone(expected_A_override)

  def test_impurity_api_warning(self):
    with self.assertLogs(level='WARNING') as log_output:
      plasma_composition.PlasmaComposition(
          impurity={
              'impurity_mode': plasma_composition._IMPURITY_MODE_FRACTIONS,
              'species': 'Ne',
              'Z_override': 5.0,
          },
          Z_impurity_override=6.0,
      )
      self.assertIn(
          'Z_impurity_override and/or A_impurity_override are set',
          log_output[0][0].message,
      )

  def test_zeff_usage_warning_with_ne_ratios(self):
    """Tests warning when Z_eff is provided with n_e_ratios impurity mode."""
    with self.assertLogs(level='WARNING') as log_output:
      plasma_composition.PlasmaComposition(
          impurity={
              'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS,
              'species': {'Ne': 0.01},
          },
          Z_eff=1.5,
      )
      self.assertIn(
          'Z_eff is provided but impurity_mode is'
          f" '{plasma_composition._IMPURITY_MODE_NE_RATIOS}'",
          log_output[0][0].message,
      )

  def test_ne_ratios_validation_error_for_negative_ratio(self):
    """Tests that n_e_ratios must be non-negative."""
    with self.assertRaises(pydantic.ValidationError):
      plasma_composition.PlasmaComposition(
          impurity={
              'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS,
              'species': {'Ne': -0.1},
          }
      )

  def test_ne_ratios_Z_eff_validation_error_for_negative_ratio(self):
    """Tests that n_e_ratios must be non-negative."""
    with self.assertRaises(pydantic.ValidationError):
      plasma_composition.PlasmaComposition(
          impurity={
              'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF,
              'species': {'Ne': -0.1, 'W': None},
          }
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_none',
          species={'Ne': 0.01, 'W': 1e-5},
          should_raise=True,
      ),
      dict(
          testcase_name='one_none',
          species={'Ne': 0.01, 'W': None},
          should_raise=False,
      ),
      dict(
          testcase_name='two_nones',
          species={'Ne': None, 'W': None},
          should_raise=True,
      ),
  )
  def test_ne_ratios_Z_eff_validation(self, species, should_raise):
    """Tests that NeRatiosZeffModel must have exactly one None species."""
    config = {
        'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF,
        'species': species,
    }
    if should_raise:
      with self.assertRaises(pydantic.ValidationError):
        plasma_composition.PlasmaComposition(impurity=config)
    else:
      plasma_composition.PlasmaComposition(impurity=config)

  @parameterized.named_parameters(
      dict(
          testcase_name='fractions',
          impurity_mode=plasma_composition._IMPURITY_MODE_FRACTIONS,
      ),
      dict(
          testcase_name='n_e_ratios',
          impurity_mode=plasma_composition._IMPURITY_MODE_NE_RATIOS,
      ),
      dict(
          testcase_name='n_e_ratios_Z_eff',
          impurity_mode=plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF,
      ),
  )
  def test_empty_species_validation(self, impurity_mode):
    """Tests validation error for empty species dict."""
    with self.assertRaises(pydantic.ValidationError):
      plasma_composition.PlasmaComposition(
          impurity={'impurity_mode': impurity_mode, 'species': {}}
      )

  def test_ne_ratios_avg_a_calculation(self):
    # These n_e_ratios correspond to 1/3 C and 2/3 N fractions.
    n_e_ratios_species = {'C': 0.01, 'N': 0.02}
    fractions_species = {'C': 1 / 3, 'N': 2 / 3}
    t = 0.0

    pc_ne_ratios = plasma_composition.PlasmaComposition(
        impurity={
            'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS,
            'species': n_e_ratios_species,
        }
    )
    ne_params = pc_ne_ratios.impurity.build_runtime_params(t)
    assert isinstance(
        ne_params, electron_density_ratios.RuntimeParams
    )

    pc_fractions = plasma_composition.PlasmaComposition(
        impurity={
            'impurity_mode': plasma_composition._IMPURITY_MODE_FRACTIONS,
            'species': fractions_species,
        }
    )
    fractions_params = pc_fractions.impurity.build_runtime_params(t)
    assert isinstance(
        fractions_params, ion_mixture.RuntimeParams
    )

    np.testing.assert_allclose(
        ne_params.A_avg,
        fractions_params.A_avg,
        rtol=1e-5,
    )

  def test_ne_ratios_model_under_jit_smoke_test(self):
    pc = plasma_composition.PlasmaComposition(
        impurity={
            'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS,
            'species': {'C': {0.0: 0.01}, 'N': {0.0: 0.02}},
        }
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(pc, geo.torax_mesh)

    @jax.jit
    def f(pc_model: plasma_composition.PlasmaComposition, t: chex.Numeric):
      return pc_model.build_runtime_params(t=t)

    # Just a smoke test to ensure it jits and runs.
    output = f(pc, 0.0)
    self.assertIsInstance(
        output.impurity, electron_density_ratios.RuntimeParams
    )
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)
    # run again to check for re-compilation
    f(pc, 0.0)
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  def test_ne_ratios_zeff_model_under_jit(self):
    """Smoke test for JIT compilation of NeRatiosZeffModel."""
    pc = plasma_composition.PlasmaComposition(
        impurity={
            'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF,
            'species': {'C': {0.0: 0.01}, 'N': None},
        },
        Z_eff=2.0,
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(pc, geo.torax_mesh)

    @jax.jit
    def f(pc_model: plasma_composition.PlasmaComposition, t: chex.Numeric):
      return pc_model.build_runtime_params(t=t)

    # Just a smoke test to ensure it jits and runs.
    output = f(pc, 0.0)
    self.assertIsInstance(
        output.impurity, electron_density_ratios_zeff.RuntimeParams
    )
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)
    # run again to check for re-compilation
    f(pc, 0.0)
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  def test_update_fields_with_legacy_impurity_input(self):
    """Tests updating legacy impurity format via update_fields."""
    config_dict = {
        'profile_conditions': {},
        'plasma_composition': {'impurity': {'Ne': 0.99, 'W': 0.01}},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    config_updates = {'plasma_composition.impurity': {'Ne': 0.98, 'W': 0.02}}

    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertEqual(
        torax_config.plasma_composition.get_impurity_names(), ('Ne', 'W')
    )
    assert torax_config.plasma_composition.impurity.species['Ne'] is not None
    assert torax_config.plasma_composition.impurity.species['W'] is not None
    self.assertEqual(
        torax_config.plasma_composition.impurity.species['Ne'].get_value(0.0),
        0.99,
    )
    self.assertEqual(
        torax_config.plasma_composition.impurity.species['W'].get_value(0.0),
        0.01,
    )
    torax_config.update_fields(config_updates)
    self.assertEqual(
        torax_config.plasma_composition.get_impurity_names(), ('Ne', 'W')
    )
    self.assertEqual(
        torax_config.plasma_composition.impurity.species['Ne'].get_value(0.0),
        0.98,
    )
    self.assertEqual(
        torax_config.plasma_composition.impurity.species['W'].get_value(0.0),
        0.02,
    )


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
      ('invalid_ion_symbol', {'De': 0.5, 'Tr': 0.5}, True),
  )
  def test_ion_mixture_constructor(self, input_species, should_raise):
    """Tests various cases of IonMixture construction."""

    if should_raise:
      with self.assertRaises(pydantic.ValidationError):
        ion_mixture.IonMixture.model_validate({'species': input_species})
    else:
      ion_mixture.IonMixture.model_validate({'species': input_species})

  # pylint: disable = invalid-name

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
          7.428571428571429,
          14.05325,
      ),
  )
  def test_ion_mixture_averaging(self, species, time, expected_Z, expected_A):
    """Tests the averaging of Z and A for different mixtures."""
    mixture = ion_mixture.IonMixture.model_validate(
        dict(species=species)
    )
    mixture_params = mixture.build_runtime_params(time)
    calculated_Z = charge_states.get_average_charge_state(
        ion_symbols=tuple(species.keys()),  # pytype: disable=attribute-error
        ion_mixture=mixture_params,
        T_e=np.array([10.0]),  # Ensure that all ions in test are fully ionized
    ).Z_mixture
    np.testing.assert_allclose(calculated_Z, expected_Z)
    np.testing.assert_allclose(mixture_params.A_avg, expected_A)

  @parameterized.named_parameters(
      ('no_override', None, None, 1.0, 2.0141),
      ('Z_override', 3.0, None, 1.0, 2.0141),
      ('A_override', None, 3.0, 1.0, 2.0141),
      ('both_override', 3.0, 3.0, 1.0, 2.0141),
  )
  def test_ion_mixture_override(self, Z_override, A_override, Z, A):
    """Tests overriding the automatic Z/A averaging."""

    mixture = ion_mixture.IonMixture.model_validate(
        dict(
            species={'D': {0: 1.0}},
            Z_override=Z_override,
            A_override=A_override,
        )
    )
    mixture_params = mixture.build_runtime_params(t=0.0)
    calculated_Z = charge_states.get_average_charge_state(
        ion_symbols=tuple(mixture.species.keys()),
        ion_mixture=mixture_params,
        T_e=np.array([1.0]),  # arbitrary temperature, won't be used for D
    ).Z_mixture
    Z_expected = Z if Z_override is None else Z_override
    A_expected = A if A_override is None else A_override
    np.testing.assert_allclose(calculated_Z, Z_expected)
    np.testing.assert_allclose(mixture_params.A_avg, A_expected)

  def test_model_validate(self):
    """Test that IonMixture.from_config behaves as expected."""
    # Single ion.
    mixture = ion_mixture.IonMixture.model_validate({'species': 'D'})
    self.assertEqual(
        mixture.species,
        {'D': torax_pydantic.TimeVaryingScalar.model_validate(1.0)},
    )

    # Multiple ions.
    mixture = ion_mixture.IonMixture.model_validate(
        dict(species={'D': 0.6, 'T': 0.4})
    )
    self.assertEqual(set(mixture.species.keys()), {'D', 'T'})

    # Check overrides.
    z = torax_pydantic.TimeVaryingScalar.model_validate(1.2)
    a = torax_pydantic.TimeVaryingScalar.model_validate(2.4)
    mixture = ion_mixture.IonMixture.model_validate(
        dict(species='D', Z_override=1.2, A_override=2.4)
    )
    self.assertEqual(mixture.Z_override, z)
    self.assertEqual(mixture.A_override, a)

  # pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
