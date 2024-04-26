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

"""Tests for sources/formulas.py."""

from absl.testing import absltest
import chex
from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.sources import default_sources
from torax.sources import formula_config
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.stepper import linear_theta_method
from torax.tests.test_lib import sim_test_case
from torax.transport_model import constant as constant_transport_model


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class FormulasIntegrationTest(sim_test_case.SimTestCase):
  """Integration tests for using non-default formulas."""

  def test_custom_exponential_source_can_replace_puff_source(self):
    """Replaces one the default ne source with a custom one."""
    # The default puff source gives an exponential profile. In this test, we
    # zero out the default puff source and introduce a new custom source that
    # should give the same profiles throughout the entire simulation run as the
    # original puff source.

    # For this test, use test_particle_sources_constant with the linear stepper.
    custom_source_name = 'custom_exponential_source'

    # Copy the test_particle_sources_constant config in here for clarity.
    test_particle_sources_constant_config = config_lib.Config(
        profile_conditions=config_lib.ProfileConditions(
            set_pedestal=True,
            nbar=0.85,
            nu=0,
        ),
        numerics=config_lib.Numerics(
            ion_heat_eq=True,
            el_heat_eq=True,
            dens_eq=True,  # This is important to be True to test ne sources.
            current_eq=True,
            resistivity_mult=100,
            t_final=2,
        ),
    )
    # Set the sources to match test_particle_sources_constant as well.
    source_models = default_sources.get_default_sources()
    source_models.sources['pellet_source'].runtime_params.S_pellet_tot = 2.0e22
    S_puff_tot = 1.0e22  # pylint: disable=invalid-name
    puff_decay_length = 0.05
    source_models.sources['gas_puff_source'].runtime_params.S_puff_tot = (
        S_puff_tot
    )
    source_models.sources[
        'gas_puff_source'
    ].runtime_params.puff_decay_length = puff_decay_length
    source_models.sources['nbi_particle_source'].runtime_params.S_nbi_tot = 0.0
    # We need to turn off some other sources for test_particle_sources_constant
    # that are unrelated to our test for the ne custom source.
    source_models.sources['fusion_heat_source'].runtime_params.mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models.sources['ohmic_heat_source'].runtime_params.mode = (
        runtime_params_lib.Mode.ZERO
    )

    # Add the custom source to the source_models, but keep it turned off for the
    # first run.
    source_models.add_source(
        custom_source_name,
        source.SingleProfileSource(
            supported_modes=(
                runtime_params_lib.Mode.ZERO,
                runtime_params_lib.Mode.FORMULA_BASED,
            ),
            affected_core_profiles=(source.AffectedCoreProfile.NE,),
            formula=formulas.Exponential(),
            runtime_params=runtime_params_lib.RuntimeParams(
                mode=runtime_params_lib.Mode.ZERO,
                # will override these later, but defining here because, due to
                # how JAX works, this function is still evaluated even when the
                # mode is set to ZERO. So the runtime config needs to be set
                # with the correct params.
                formula=formula_config.Exponential(),
            ),
        ),
    )

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(
        'test_particle_sources_constant.h5', _ALL_PROFILES
    )

    # We set up the sim only once and update the config on each run below in a
    # way that does not trigger recompiles. This way we only trace the code
    # once.
    geo = geometry.build_circular_geometry(
        test_particle_sources_constant_config
    )
    transport_model = constant_transport_model.ConstantTransportModel(
        runtime_params=constant_transport_model.RuntimeParams(
            De_const=0.5,
            Ve_const=-0.2,
        )
    )
    sim = sim_lib.build_sim_from_config(
        config=test_particle_sources_constant_config,
        geo=geo,
        stepper_builder=linear_theta_method.LinearThetaMethodBuilder(
            runtime_params=linear_theta_method.LinearRuntimeParams(
                predictor_corrector=False,
            )
        ),
        transport_model=transport_model,
        source_models=source_models,
    )

    # Make sure the config copied here works with these references.
    with self.subTest('with_puff_and_without_custom_source'):
      # Need to run the sim once to build the step_fn.
      torax_outputs = sim.run()
      state_history, _, _ = state_lib.build_history_from_states(torax_outputs)
      t = state_lib.build_time_history_from_states(torax_outputs)
      self._check_profiles_vs_expected(
          state_history=state_history,
          t=t,
          ref_time=ref_time,
          ref_profiles=ref_profiles,
          rtol=self.rtol,
          atol=self.atol,
      )

    with self.subTest('without_puff_and_with_custom_source'):
      # Now turn on the custom source.
      source_models.sources[custom_source_name].runtime_params.mode = (
          runtime_params_lib.Mode.FORMULA_BASED
      )
      source_models.sources[custom_source_name].runtime_params.formula = (
          formula_config.Exponential(
              total=(
                  S_puff_tot
                  / test_particle_sources_constant_config.numerics.nref
              ),
              c1=1.0,
              c2=puff_decay_length,
              use_normalized_r=True,
          )
      )
      # And turn off the gas puff source it is replacing.
      source_models.sources['gas_puff_source'].runtime_params.mode = (
          runtime_params_lib.Mode.ZERO
      )
      self._run_sim_and_check(sim, ref_profiles, ref_time)

    with self.subTest('without_puff_and_without_custom_source'):
      # Confirm that the custom source actual has an effect.
      # Turn it off as well, and the check shouldn't pass.
      source_models.sources[custom_source_name].runtime_params.mode = (
          runtime_params_lib.Mode.ZERO
      )
      with self.assertRaises(AssertionError):
        self._run_sim_and_check(sim, ref_profiles, ref_time)

  def _run_sim_and_check(
      self,
      sim: sim_lib.Sim,
      ref_profiles: dict[str, chex.ArrayTree],
      ref_time: chex.Array,
  ):
    """Runs sim with new dynamic config and checks the profiles vs. expected."""
    torax_outputs = sim_lib.run_simulation(
        static_config_slice=sim.static_config_slice,
        dynamic_config_slice_provider=sim.dynamic_config_slice_provider,
        geometry_provider=sim.geometry_provider,
        initial_state=sim.initial_state,
        time_step_calculator=sim.time_step_calculator,
        step_fn=sim.step_fn,
    )
    state_history, _, _ = state_lib.build_history_from_states(torax_outputs)
    t = state_lib.build_time_history_from_states(torax_outputs)
    self._check_profiles_vs_expected(
        state_history=state_history,
        t=t,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=self.rtol,
        atol=self.atol,
    )


if __name__ == '__main__':
  absltest.main()
