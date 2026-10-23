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
import dataclasses
from typing import ClassVar
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.config import numerics
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import electron_cyclotron_source
from torax._src.sources import generic_current_source
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_state as source_state_lib


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_missing_source_id_raises_error(self):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.Source):
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[source_lib.AffectedCoreProfile, ...]
      ] = (source_lib.AffectedCoreProfile.PSI,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(ValueError, 'Source ID must be set.'):
      BadSource()

  def test_missing_affected_core_profiles_raises_error(self):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.Source):
      SOURCE_ID: ClassVar[str] = 'bad'

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(
        ValueError, 'Affected core profiles must be set.'
    ):
      BadSource()

  def test_zero_profile_works_by_default(self):
    """The default source impl should support profiles with all zeros."""
    source = generic_current_source.GenericCurrentSource()
    geo = mock.create_autospec(
        geometry.Geometry, rho_norm=np.array([1, 1, 1, 1])
    )
    source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=np.zeros_like(geo.rho_norm),
                mode=sources_runtime_params_lib.Mode.ZERO,
                is_explicit=False,
            )
        )
    }
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=source_params,
    )
    profile, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source_state_lib.SourceState(),
    )
    np.testing.assert_allclose(profile[0], np.zeros_like(geo.rho_norm))  # pyrefly: ignore[no-matching-overload]

  @parameterized.parameters(
      (sources_runtime_params_lib.Mode.ZERO, np.array([0, 0, 0, 0])),
      (
          sources_runtime_params_lib.Mode.MODEL_BASED,
          np.array([42, 42, 42, 42]),
      ),
      (sources_runtime_params_lib.Mode.PRESCRIBED, np.array([3, 3, 3, 3])),
  )
  def test_correct_mode_called(
      self,
      mode,
      expected_profile,
  ):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class MockGenericCurrentSource(generic_current_source.GenericCurrentSource):

      def _get_model_value(self, *args, **kwargs):
        return (np.full([4], 42.0),)

    source = MockGenericCurrentSource()
    dynamic_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=(np.full([4], 3.0),),
                mode=mode,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
        numerics=mock.create_autospec(
            numerics.RuntimeParams, enable_fast_ions=True
        ),
    )
    # Make a geo with rho_norm as we need it for the zero profile shape.
    geo = mock.create_autospec(
        geometry.Geometry, rho_norm=np.array([1, 1, 1, 1])
    )
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=geo,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source_state_lib.SourceState(),
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        profile[0],
        expected_profile,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_prescribed_values_for_multiple_affected_profiles(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=(np.full([4], 3.0), np.full([4], 4.0)),
                mode=sources_runtime_params_lib.Mode.PRESCRIBED,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
        numerics=mock.create_autospec(
            numerics.RuntimeParams, enable_fast_ions=True
        ),
    )
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=mock.ANY,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source_state_lib.SourceState(),
    )
    self.assertLen(profile, 2)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        profile[0],
        np.full([4], 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        profile[1],
        np.full([4], 4.0),
        atol=1e-6,
        rtol=1e-6,
    )

  def test_source_with_mismatched_prescribed_values_raises_error(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=(np.full([4], 3.0),),
                mode=sources_runtime_params_lib.Mode.PRESCRIBED,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
        numerics=mock.create_autospec(
            numerics.RuntimeParams, enable_fast_ions=True
        ),
    )
    with self.assertRaisesRegex(
        ValueError,
        'the number of prescribed values must match the number of affected',
    ):
      source.get_value(
          runtime_params=dynamic_slice,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          calculated_source_profiles=None,
          conductivity=None,
          source_state=source_state_lib.SourceState(),
      )

  def test_fast_ions_populates_missing_species_and_validates(self):
    zero_d = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='D'
    )
    zero_he3 = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='He3'
    )
    computed_he3 = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='He3'
    )

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class DummyFastIonSource(source_lib.Source):
      SOURCE_ID: ClassVar[str] = 'dummy_fast_ion'
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[source_lib.AffectedCoreProfile, ...]
      ] = (source_lib.AffectedCoreProfile.FAST_IONS,)
      fast_ions_out: tuple[fast_ion_lib.FastIon, ...] = ()

      def _get_model_value(self, *args, **kwargs):
        return (self.fast_ions_out,)

      def zero_fast_ions(self, geo):
        del geo
        return (zero_d, zero_he3)

    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            'dummy_fast_ion': sources_runtime_params_lib.RuntimeParams(
                prescribed_values=(),
                mode=sources_runtime_params_lib.Mode.MODEL_BASED,
                is_explicit=False,
            )
        },
        numerics=mock.create_autospec(
            numerics.RuntimeParams, enable_fast_ions=True
        ),
    )

    source = DummyFastIonSource(fast_ions_out=(computed_he3,))
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=mock.ANY,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source_state_lib.SourceState(),
    )
    self.assertEqual(profile[0], (zero_d, computed_he3))

    # Unsupported species raises ValueError.
    bad_species = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='W'
    )
    bad_source = DummyFastIonSource(fast_ions_out=(bad_species,))
    with self.assertRaisesRegex(ValueError, 'Unsupported FastIon species'):
      bad_source.get_value(
          runtime_params=dynamic_slice,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          calculated_source_profiles=None,
          conductivity=None,
          source_state=source_state_lib.SourceState(),
      )

    # Duplicate species raises ValueError.
    dup_source = DummyFastIonSource(
        fast_ions_out=(computed_he3, computed_he3)
    )
    with self.assertRaisesRegex(ValueError, 'Duplicate FastIon species'):
      dup_source.get_value(
          runtime_params=dynamic_slice,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          calculated_source_profiles=None,
          conductivity=None,
          source_state=source_state_lib.SourceState(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='fast_ions_enabled',
          enable_fast_ions=True,
          expected_species=('D', 'He3'),
      ),
      dict(
          testcase_name='fast_ions_disabled',
          enable_fast_ions=False,
          expected_species=(),
      ),
  )
  def test_prescribed_values_omitting_fast_ions(
      self,
      enable_fast_ions: bool,
      expected_species: tuple[str, ...],
  ):
    zero_d = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='D'
    )
    zero_he3 = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species='He3'
    )
    species_map = {'D': zero_d, 'He3': zero_he3}

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class DummyHeatingAndFastIonSource(source_lib.Source):
      SOURCE_ID: ClassVar[str] = 'dummy_icrh'
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[source_lib.AffectedCoreProfile, ...]
      ] = (
          source_lib.AffectedCoreProfile.TEMP_ION,
          source_lib.AffectedCoreProfile.TEMP_EL,
          source_lib.AffectedCoreProfile.FAST_IONS,
      )

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

      def zero_fast_ions(self, geo):
        del geo
        return (zero_d, zero_he3)

    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            'dummy_icrh': sources_runtime_params_lib.RuntimeParams(
                prescribed_values=(np.full([4], 3.0), np.full([4], 4.0)),
                mode=sources_runtime_params_lib.Mode.PRESCRIBED,
                is_explicit=False,
            )
        },
        numerics=mock.create_autospec(
            numerics.RuntimeParams, enable_fast_ions=enable_fast_ions
        ),
    )

    source = DummyHeatingAndFastIonSource()
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=mock.ANY,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source_state_lib.SourceState(),
    )

    self.assertLen(profile, 3)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        profile[0],
        np.full([4], 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        profile[1],
        np.full([4], 4.0),
        atol=1e-6,
        rtol=1e-6,
    )
    self.assertEqual(
        profile[2],
        tuple(species_map[s] for s in expected_species),
    )


if __name__ == '__main__':
  absltest.main()
