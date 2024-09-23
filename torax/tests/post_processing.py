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

"""Tests for post_processing.py."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax import geometry
from torax import geometry_provider
from torax import post_processing
from torax import state
from torax.config import runtime_params as runtime_params_lib
from torax.fvm import cell_variable
from torax.sources import default_sources
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import torax_refs


class PostProcessingTest(parameterized.TestCase):
  """Unit tests for the `post_processing` module."""

  def setUp(self):
    super().setUp()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    self.geo = geometry.build_circular_geometry()
    geo_provider = geometry_provider.ConstantGeometryProvider(self.geo)
    source_models_builder = default_sources.get_default_sources_builder()
    source_models = source_models_builder()
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=source_models_builder.runtime_params,
        )
    )
    # Make some dummy source profiles.
    ones = np.ones_like(geo.rho)
    self.source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        profiles={
            'bremsstrahlung_heat_sink': -ones,
            'ohmic_heat_source': ones * 5,
        },
    )
    self.core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

  def test_make_outputs(self):
    """Test that post-processing outputs are added to the state."""
    sim_state = state.ToraxSimState(
        core_profiles=self.core_profiles,
        core_transport=state.CoreTransport.zeros(self.geo),
        core_sources=self.source_profiles,
        t=jax.numpy.array(0.0),
        dt=jax.numpy.array(0.1),
        time_step_calculator_state=None,
        post_processed_outputs=state.PostProcessedOutputs.zeros(self.geo),
        stepper_numeric_outputs=state.StepperNumericOutputs(
            outer_stepper_iterations=1,
            stepper_error_state=1,
            inner_solver_iterations=1,
        ),
    )

    updated_sim_state = post_processing.make_outputs(sim_state, self.geo)

    # Check that the outputs were updated.
    for field in state.PostProcessedOutputs.__dataclass_fields__:
      with self.subTest(field=field):
        try:
          np.testing.assert_array_equal(
              getattr(updated_sim_state.post_processed_outputs, field),
              getattr(sim_state.post_processed_outputs, field),
          )
        except AssertionError:
          # At least one field is different, so the test passes.
          return
    # If no assertion error was raised, then all fields are the same
    # so raise an error.
    raise AssertionError('PostProcessedOutputs did not change.')

  def test_compute_pressure(self):
    """Test that pressure is computed correctly."""

    def _make_constant_core_profile(
        value: float,
    ) -> cell_variable.CellVariable:
      return cell_variable.CellVariable(
          value=value * np.ones_like(self.geo.rho_norm),
          left_face_grad_constraint=np.zeros(()),
          left_face_constraint=None,
          right_face_grad_constraint=None,
          right_face_constraint=jax.numpy.array(value),
          dr=self.geo.drho_norm,
      )

    # Override all densities and temperatures to constant values
    core_profiles = dataclasses.replace(
        self.core_profiles,
        temp_ion=_make_constant_core_profile(1.0),
        temp_el=_make_constant_core_profile(2.0),
        ne=_make_constant_core_profile(3.0),
        ni=_make_constant_core_profile(2.5),
        nimp=_make_constant_core_profile(0.25),
    )
    # pylint: disable=protected-access
    p_el, p_ion, p_tot = post_processing._compute_pressure(core_profiles)
    # pylint: enable=protected-access
    # Make sure that we are grabbing the values from the face grid.
    self.assertEqual(p_el.shape, self.geo.rho_face.shape)
    # Ignore boundary condition terms and just check formula sanity.
    np.testing.assert_allclose(
        p_el, 6 * constants.CONSTANTS.keV2J * core_profiles.nref
    )
    np.testing.assert_allclose(
        p_ion,
        2.75 * constants.CONSTANTS.keV2J * core_profiles.nref,
    )
    np.testing.assert_allclose(
        p_tot,
        8.75 * constants.CONSTANTS.keV2J * core_profiles.nref,
    )

  def test_compute_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    geo = geometry.build_circular_geometry()
    p_el = np.ones_like(geo.rho_face)
    p_ion = 2 * np.ones_like(geo.rho_face)
    p_tot = p_el + p_ion
    # pylint: disable=protected-access
    wth_el, wth_ion, wth_tot = post_processing._compute_stored_thermal_energy(
        p_el, p_ion, p_tot, geo
    )
    # pylint: enable=protected-access

    volume = np.trapz(geo.vpr_face, geo.rho_face_norm)

    np.testing.assert_allclose(wth_el, 1.5 * p_el[0] * volume)
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion[0] * volume)
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot[0] * volume)


if __name__ == '__main__':
  absltest.main()
