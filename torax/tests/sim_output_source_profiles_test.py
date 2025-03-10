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

"""Tests checking the output core_sources profiles from run_simulation().

This is a separate file to not bloat the main sim.py test file.
"""

from __future__ import annotations

import dataclasses
from unittest import mock

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from torax import sim as sim_lib
from torax import state as state_module
from torax.config import runtime_params as general_runtime_params
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.orchestration import step_function
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib
from torax.sources.tests import test_lib
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import explicit_stepper
from torax.tests.test_lib import sim_test_case
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import constant as constant_transport_model


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class TestImplicitNeSource(test_lib.TestSource):
  """A test source."""

  @property
  def source_name(self) -> str:
    return 'implicit_ne_source'


class TestExplicitNeSource(test_lib.TestSource):
  """A test source."""

  @property
  def source_name(self) -> str:
    return 'explicit_ne_source'


class SimOutputSourceProfilesTest(sim_test_case.SimTestCase):
  """Tests checking the output core_sources profiles from run_simulation()."""

  def test_merging_source_profiles(self):
    """Tests that the implicit and explicit source profiles merge correctly."""
    torax_mesh = geometry.Grid1D.construct(10, 0.1)
    source_models_builder = default_sources.get_default_sources_builder()
    source_models = source_models_builder()
    # Technically, the merge_source_profiles() function should be called with
    # source profiles where, for every source, only one of the implicit or
    # explicit profiles has non-zero values. That is what makes the summing
    # correct. For this test though, we are simply checking that things are
    # summed in the first place.
    # Build a fake set of source profiles which have all 1s in all the profiles.
    fake_implicit_source_profiles = _build_source_profiles_with_single_value(
        torax_mesh=torax_mesh,
        source_models=source_models,
        value=1.0,
    )
    # And a fake set of profiles with all 2s.
    fake_explicit_source_profiles = _build_source_profiles_with_single_value(
        torax_mesh=torax_mesh,
        source_models=source_models,
        value=2.0,
    )
    merged_profiles = source_profiles_lib.SourceProfiles.merge(
        implicit_source_profiles=fake_implicit_source_profiles,
        explicit_source_profiles=fake_explicit_source_profiles,
    )
    # All the profiles in the merged profiles should be a 1D array with all 3s.
    for profile in merged_profiles.temp_el.values():
      np.testing.assert_allclose(profile, 3.0)
    for profile in merged_profiles.temp_ion.values():
      np.testing.assert_allclose(profile, 3.0)
    for profile in merged_profiles.psi.values():
      np.testing.assert_allclose(profile, 3.0)
    for profile in merged_profiles.ne.values():
      np.testing.assert_allclose(profile, 3.0)
    np.testing.assert_allclose(merged_profiles.qei.qei_coef, 3.0)
    # Make sure the combo ion-el heat sources are present.
    for name in ['generic_ion_el_heat_source', 'fusion_heat_source']:
      self.assertIn(name, merged_profiles.temp_ion)
      self.assertIn(name, merged_profiles.temp_el)

  def test_first_and_last_source_profiles(self):
    """Tests that the first and last source profiles contain correct data."""
    # The first time step and last time step's output source profiles are built
    # in a special way that combines the implicit and explicit profiles.

    # Create custom sources whose output profiles depend on foo.
    # This is not physically realistic, just for testing purposes.
    def custom_source_formula(
        unused_static_runtime_params_slice,
        dynamic_runtime_params,
        unused_geo,
        source_name,
        unused_state,
        unused_calculated_source_profiles,
    ):
      dynamic_source_params = dynamic_runtime_params.sources[source_name]
      return (dynamic_source_params.prescribed_values,)

    # Include 2 versions of this source, one implicit and one explicit.
    runtime_params = runtime_params_lib.RuntimeParams(
        mode=runtime_params_lib.Mode.MODEL_BASED,
        prescribed_values={
            0.0: {0: 1.0},
            1.0: {0: 2.0},
            2.0: {0: 3.0},
            3.0: {0: 4.0},
        },
    )
    implicit_source_builder = source_lib.make_source_builder(
        TestImplicitNeSource,
        runtime_params_type=runtime_params_lib.RuntimeParams,
        model_func=custom_source_formula,
    )
    explicit_source_builder = source_lib.make_source_builder(
        TestExplicitNeSource,
        runtime_params_type=runtime_params_lib.RuntimeParams,
        model_func=custom_source_formula,
    )
    source_models_builder = source_models_lib.SourceModelsBuilder({
        'implicit_ne_source': implicit_source_builder(
            runtime_params=runtime_params,
        ),
        'explicit_ne_source': explicit_source_builder(
            runtime_params=runtime_params,
        ),
    })
    source_models = source_models_builder()
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    runtime_params.numerics.t_final = 2.0
    runtime_params.numerics.fixed_dt = 1.0
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    time_stepper = fixed_time_step_calculator.FixedTimeStepCalculator()

    def mock_step_fn(
        _,
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geometry_provider,
        input_state,
    ):
      dt = 1.0
      new_t = input_state.t + dt
      return (
          dataclasses.replace(
              input_state,
              t=new_t,
              dt=dt,
              time_step_calculator_state=(),
              core_sources=source_profile_builders.get_initial_source_profiles(
                  static_runtime_params_slice,
                  dynamic_runtime_params_slice_provider(new_t),
                  geometry_provider(new_t),
                  core_profiles=input_state.core_profiles,
                  source_models=source_models,
              ),
          ),
          state_module.SimError.NO_ERROR,
      )

    sim = sim_lib.Sim.create(
        runtime_params=runtime_params,
        geometry_provider=geometry_provider_lib.ConstantGeometryProvider(geo),
        stepper=explicit_stepper.ExplicitStepperModel(),
        transport_model_builder=constant_transport_model.ConstantTransportModelBuilder(),
        source_models_builder=source_models_builder,
        pedestal=pedestal_pydantic_model.Pedestal(),
        time_step_calculator=time_stepper,
    )
    with mock.patch.object(
        step_function.SimulationStepFn, '__call__', new=mock_step_fn
    ):
      sim_outputs = sim.run()

    # The implicit and explicit profiles get merged together before being
    # outputted, and they are aligned as well as possible to be computed based
    # on the state and config at time t. So both the implicit and explicit
    # profiles of each time step should be equal in this case (especially
    # because we are using the mock step function defined above).
    for i, sim_state in enumerate(sim_outputs.sim_history):
      np.testing.assert_allclose(
          sim_state.core_sources.ne['implicit_ne_source'], i + 1
      )
      np.testing.assert_allclose(
          sim_state.core_sources.ne['explicit_ne_source'], i + 1
      )


def _build_source_profiles_with_single_value(
    torax_mesh: geometry.Grid1D,
    source_models: source_models_lib.SourceModels,
    value: float,
) -> source_profiles_lib.SourceProfiles:
  """Builds a set of source profiles with all values set to a single value."""
  cell_1d_arr = jnp.full((torax_mesh.nx,), value)
  face_1d_arr = jnp.full((torax_mesh.nx + 1), value)
  profiles = {
      source_lib.AffectedCoreProfile.PSI: {},
      source_lib.AffectedCoreProfile.NE: {},
      source_lib.AffectedCoreProfile.TEMP_ION: {},
      source_lib.AffectedCoreProfile.TEMP_EL: {},
  }
  for source_name, source in source_models.standard_sources.items():
    for affected_core_profile in source.affected_core_profiles:
      profiles[affected_core_profile][source_name] = cell_1d_arr
  return source_profiles_lib.SourceProfiles(
      temp_el=profiles[source_lib.AffectedCoreProfile.TEMP_EL],
      temp_ion=profiles[source_lib.AffectedCoreProfile.TEMP_ION],
      ne=profiles[source_lib.AffectedCoreProfile.NE],
      psi=profiles[source_lib.AffectedCoreProfile.PSI],
      j_bootstrap=source_profiles_lib.BootstrapCurrentProfile(
          sigma=cell_1d_arr,
          sigma_face=face_1d_arr,
          j_bootstrap=cell_1d_arr,
          j_bootstrap_face=face_1d_arr,
          I_bootstrap=jnp.ones(()) * value,
      ),
      qei=source_profiles_lib.QeiInfo(
          qei_coef=cell_1d_arr,
          implicit_ii=cell_1d_arr,
          explicit_i=cell_1d_arr,
          implicit_ee=cell_1d_arr,
          explicit_e=cell_1d_arr,
          implicit_ie=cell_1d_arr,
          implicit_ei=cell_1d_arr,
      ),
  )


if __name__ == '__main__':
  absltest.main()
