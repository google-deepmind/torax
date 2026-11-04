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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import core_sources
from torax._src.sources import electron_cyclotron_source
from torax._src.sources import fusion_heat_source
from torax._src.sources import generic_current_source
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_missing_source_id_raises_error(self):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.Source):
      POPULATED_SOURCE_OUTPUTS: ClassVar[
          tuple[core_sources.PopulatedSourceOutput, ...]
      ] = (core_sources.PopulatedSourceOutput.J_PARALLEL,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(ValueError, "Source ID must be set."):
      BadSource()

  def test_missing_populated_source_outputs_raises_error(self):

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.Source):
      SOURCE_ID: ClassVar[str] = "bad"

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(
        ValueError, "Populated source outputs must be set."
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
                prescribed_values=core_sources.SourceOutput(
                    j_parallel=np.zeros_like(geo.rho_norm)
                ),
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
        solver_profiles=mock.ANY,
        source_state=source_state_lib.SourceState(),
    )
    assert profile.j_parallel is not None
    np.testing.assert_allclose(profile.j_parallel, np.zeros_like(geo.rho_norm))

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
        source_state = kwargs.get(
            "source_state",
            args[-1] if args else source_state_lib.SourceState(),
        )
        return (
            core_sources.SourceOutput(j_parallel=np.full([4], 42.0)),
            source_state,
        )

    source = MockGenericCurrentSource()
    dynamic_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=core_sources.SourceOutput(
                    j_parallel=np.full([4], 3.0)
                ),
                mode=mode,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
    )
    # Make a geo with rho_norm as we need it for the zero profile shape.
    geo = mock.create_autospec(
        geometry.Geometry, rho_norm=np.array([1, 1, 1, 1])
    )
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=geo,
        solver_profiles=mock.ANY,
        source_state=source_state_lib.SourceState(),
    )
    assert profile.j_parallel is not None
    np.testing.assert_allclose(
        profile.j_parallel,
        expected_profile,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_prescribed_values_for_multiple_populated_outputs(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=core_sources.SourceOutput(
                    power_density_e=np.full([4], 3.0),
                    j_parallel=np.full([4], 4.0),
                ),
                mode=sources_runtime_params_lib.Mode.PRESCRIBED,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
    )
    profile, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=mock.ANY,
        solver_profiles=mock.ANY,
        source_state=source_state_lib.SourceState(),
    )
    assert profile.power_density_e is not None
    assert profile.j_parallel is not None
    np.testing.assert_allclose(
        profile.power_density_e,
        np.full([4], 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        profile.j_parallel,
        np.full([4], 4.0),
        atol=1e-6,
        rtol=1e-6,
    )

  def test_source_with_mismatched_prescribed_values_raises_error(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID: (
            sources_runtime_params_lib.RuntimeParams(
                prescribed_values=core_sources.SourceOutput(
                    power_density_e=np.full([4], 3.0),
                ),
                mode=sources_runtime_params_lib.Mode.PRESCRIBED,
                is_explicit=False,
            )
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
    )
    with self.assertRaisesRegex(
        ValueError,
        r"ElectronCyclotronSource populated .* expected .*"
        r" \(POPULATED_SOURCE_OUTPUTS\)",
    ):
      source.get_value(
          runtime_params=dynamic_slice,
          geo=mock.ANY,
          solver_profiles=mock.ANY,
          source_state=source_state_lib.SourceState(),
      )

  def test_source_using_core_profiles_rejects_j_parallel(self):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.SourceUsingCoreProfiles):
      SOURCE_ID: ClassVar[str] = "bad_core_profile_source"
      POPULATED_SOURCE_OUTPUTS: ClassVar[
          tuple[core_sources.PopulatedSourceOutput, ...]
      ] = (core_sources.PopulatedSourceOutput.J_PARALLEL,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(ValueError, "cannot populate.*J_PARALLEL"):
      BadSource()

  def test_final_source_rejects_j_parallel(self):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class BadSource(source_lib.FinalSource):
      SOURCE_ID: ClassVar[str] = "bad_final_source"
      POPULATED_SOURCE_OUTPUTS: ClassVar[
          tuple[core_sources.PopulatedSourceOutput, ...]
      ] = (core_sources.PopulatedSourceOutput.J_PARALLEL,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(ValueError, "cannot populate.*J_PARALLEL"):
      BadSource()

  def test_source_using_core_profiles_rejects_fast_ions_output(self):
    dummy_fi = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species="D"
    )

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class FastIonCoreProfilesSource(source_lib.SourceUsingCoreProfiles):
      SOURCE_ID: ClassVar[str] = "fast_ion_core_source"
      POPULATED_SOURCE_OUTPUTS: ClassVar[
          tuple[core_sources.PopulatedSourceOutput, ...]
      ] = (core_sources.PopulatedSourceOutput.POWER_DENSITY_E,)

      def _get_model_value(self, *args, **kwargs):
        return (
            core_sources.SourceOutput(
                power_density_e=np.zeros(4), fast_ions=(dummy_fi,)
            ),
            source_state_lib.SourceState(),
        )

    src = FastIonCoreProfilesSource()
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            "fast_ion_core_source": sources_runtime_params_lib.RuntimeParams(
                prescribed_values=core_sources.SourceOutput(),
                mode=sources_runtime_params_lib.Mode.MODEL_BASED,
                is_explicit=False,
            )
        },
    )
    with self.assertRaisesRegex(ValueError, "cannot produce fast_ions"):
      src.get_value(
          runtime_params=runtime_params,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          source_state=source_state_lib.SourceState(),
          phi_b_dot=0.0,
      )

  def test_final_source_rejects_fast_ions_output(self):
    dummy_fi = mock.create_autospec(
        fast_ion_lib.FastIon, instance=True, species="D"
    )

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class FastIonFinalSource(source_lib.FinalSource):
      SOURCE_ID: ClassVar[str] = "fast_ion_final_source"
      POPULATED_SOURCE_OUTPUTS: ClassVar[
          tuple[core_sources.PopulatedSourceOutput, ...]
      ] = (core_sources.PopulatedSourceOutput.POWER_DENSITY_E,)

      def _get_model_value(self, *args, **kwargs):
        return (
            core_sources.SourceOutput(
                power_density_e=np.zeros(4), fast_ions=(dummy_fi,)
            ),
            source_state_lib.SourceState(),
        )

    src = FastIonFinalSource()
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={
            "fast_ion_final_source": sources_runtime_params_lib.RuntimeParams(
                prescribed_values=core_sources.SourceOutput(),
                mode=sources_runtime_params_lib.Mode.MODEL_BASED,
                is_explicit=False,
            )
        },
    )
    with self.assertRaisesRegex(ValueError, "cannot produce fast_ions"):
      src.get_value(
          runtime_params=runtime_params,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          calculated_core_sources=mock.ANY,
          source_state=source_state_lib.SourceState(),
      )

  def test_prescribed_fast_ions_populated_on_fast_ion_source(self):
    geo_config = circular_geometry.CircularConfig()
    geo = geo_config.build_runtime_params(0.0).geometry
    grid = geo_config.get_grid()
    cfg = fusion_heat_source.FusionHeatSourceConfig.from_dict({
        "mode": "prescribed",
        "prescribed_values": {
            "power_density_i": 1.0e6,
            "power_density_e": 2.0e6,
            "fast_ions": {
                "He4": {
                    "n": {0.0: {0.0: 2.0e19, 1.0: 1.0e18}},
                    "T": {0.0: {0.0: 80.0, 1.0: 5.0}},
                },
            },
        },
    })
    torax_pydantic.set_grid(cfg, grid)
    source = cfg.build_source()
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources={source.SOURCE_ID: cfg.build_runtime_params(t=0.0)},
    )
    output, _ = source.get_value(
        runtime_params=dynamic_slice,
        geo=geo,
        solver_profiles=mock.ANY,
        source_state=source_state_lib.SourceState(),
    )
    self.assertLen(output.fast_ions, 1)
    fast_ion = output.fast_ions[0]
    self.assertEqual(fast_ion.species, "He4")
    self.assertEqual(fast_ion.source, "fusion")
    expected_n = np.interp(geo.rho_norm, [0.0, 1.0], [2.0e19, 1.0e18])
    expected_t = np.interp(geo.rho_norm, [0.0, 1.0], [80.0, 5.0])
    np.testing.assert_allclose(fast_ion.n.value, expected_n)
    np.testing.assert_allclose(fast_ion.T.value, expected_t)
    self.assertIsNone(fast_ion.n.right_face_grad_constraint)
    self.assertIsNone(fast_ion.T.right_face_grad_constraint)
    assert fast_ion.n.right_face_constraint is not None
    assert fast_ion.T.right_face_constraint is not None
    np.testing.assert_allclose(fast_ion.n.right_face_constraint, 1.0e18)
    np.testing.assert_allclose(fast_ion.T.right_face_constraint, 5.0)

  def test_fast_ions_on_non_fast_ion_source_raises_error(self):
    source = generic_current_source.GenericCurrentSource()
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    fast_ion = fast_ion_lib.FastIon(
        species="D",
        source=source.SOURCE_ID,
        n=cell_variable.CellVariable(
            value=np.zeros_like(geo.rho_norm),
            face_centers=geo.rho_face_norm,
            right_face_grad_constraint=None,
            right_face_constraint=np.asarray(0.0),
        ),
        T=cell_variable.CellVariable(
            value=np.zeros_like(geo.rho_norm),
            face_centers=geo.rho_face_norm,
            right_face_grad_constraint=None,
            right_face_constraint=np.asarray(0.0),
        ),
    )
    dynamic_source_params = {
        source.SOURCE_ID: sources_runtime_params_lib.RuntimeParams(
            prescribed_values=core_sources.SourceOutput(
                j_parallel=np.zeros_like(geo.rho_norm),
                fast_ions=(fast_ion,),
            ),
            mode=sources_runtime_params_lib.Mode.PRESCRIBED,
            is_explicit=False,
        )
    }
    dynamic_slice = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        sources=dynamic_source_params,
    )
    with self.assertRaisesRegex(
        ValueError,
        "GenericCurrentSource provided fast_ions, but PRODUCES_FAST_IONS is"
        " False",
    ):
      source.get_value(
          runtime_params=dynamic_slice,
          geo=geo,
          solver_profiles=mock.ANY,
          source_state=source_state_lib.SourceState(),
      )


if __name__ == "__main__":
  absltest.main()
