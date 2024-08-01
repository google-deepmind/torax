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
from torax import output
from torax import sim as sim_lib
from torax.config import build_sim
from torax.config import runtime_params as general_runtime_params
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
    test_particle_sources_constant_runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            set_pedestal=True,
            nbar=0.85,
            nu=0,
            ne_bound_right=0.5,
        ),
        numerics=general_runtime_params.Numerics(
            ion_heat_eq=True,
            el_heat_eq=True,
            dens_eq=True,  # This is important to be True to test ne sources.
            current_eq=True,
            resistivity_mult=100,
            t_final=2,
        ),
    )
    # Set the sources to match test_particle_sources_constant as well.
    source_models_builder = default_sources.get_default_sources_builder()
    source_models_builder.runtime_params['pellet_source'].S_pellet_tot = 2.0e22
    S_puff_tot = 1.0e22  # pylint: disable=invalid-name
    puff_decay_length = 0.05
    source_models_builder.runtime_params['gas_puff_source'].S_puff_tot = (
        S_puff_tot
    )
    source_models_builder.runtime_params[
        'gas_puff_source'
    ].puff_decay_length = puff_decay_length
    source_models_builder.runtime_params['nbi_particle_source'].S_nbi_tot = 0.0
    # We need to turn off some other sources for test_particle_sources_constant
    # that are unrelated to our test for the ne custom source.
    source_models_builder.runtime_params['fusion_heat_source'].mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models_builder.runtime_params['ohmic_heat_source'].mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models_builder.runtime_params['bremsstrahlung_heat_sink'].mode = (
        runtime_params_lib.Mode.ZERO
    )

    # Add the custom source to the source_models, but keep it turned off for the
    # first run.
    source_models_builder.source_builders[custom_source_name] = (
        source.SingleProfileSourceBuilder(
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
        )
    )

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(
        'test_particle_sources_constant.nc', _ALL_PROFILES
    )

    # We set up the sim only once and update the config on each run below in a
    # way that does not trigger recompiles. This way we only trace the code
    # once.
    geo_provider = build_sim.build_geometry_provider_from_config(
        {'geometry_type': 'circular'}
    )
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                De_const=0.5,
                Ve_const=-0.2,
            )
        )
    )
    sim = sim_lib.build_sim_object(
        runtime_params=test_particle_sources_constant_runtime_params,
        geometry_provider=geo_provider,
        stepper_builder=linear_theta_method.LinearThetaMethodBuilder(
            runtime_params=linear_theta_method.LinearRuntimeParams(
                predictor_corrector=False,
            )
        ),
        transport_model_builder=transport_model_builder,
        source_models_builder=source_models_builder,
    )

    # Make sure the config copied here works with these references.
    with self.subTest('with_puff_and_without_custom_source'):
      # Need to run the sim once to build the step_fn.
      torax_outputs = sim.run()
      history = output.StateHistory(torax_outputs)
      self._check_profiles_vs_expected(
          core_profiles=history.core_profiles,
          t=history.times,
          ref_time=ref_time,
          ref_profiles=ref_profiles,
          rtol=self.rtol,
          atol=self.atol,
      )

    with self.subTest('without_puff_and_with_custom_source'):
      # Now turn on the custom source.
      source_models_builder.runtime_params[custom_source_name].mode = (
          runtime_params_lib.Mode.FORMULA_BASED
      )
      source_models_builder.runtime_params[custom_source_name].formula = (
          formula_config.Exponential(
              total=(
                  S_puff_tot
                  / test_particle_sources_constant_runtime_params.numerics.nref
              ),
              c1=1.0,
              c2=puff_decay_length,
              use_normalized_r=True,
          )
      )
      # And turn off the gas puff source it is replacing.
      source_models_builder.runtime_params['gas_puff_source'].mode = (
          runtime_params_lib.Mode.ZERO
      )
      self._run_sim_and_check(sim, ref_profiles, ref_time)

    with self.subTest('without_puff_and_without_custom_source'):
      # Confirm that the custom source actual has an effect.
      # Turn it off as well, and the check shouldn't pass.
      source_models_builder.runtime_params[custom_source_name].mode = (
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
    """Runs sim with new runtime params and checks the profiles vs. expected."""
    torax_outputs = sim_lib.run_simulation(
        static_runtime_params_slice=sim.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
        geometry_provider=sim.geometry_provider,
        initial_state=sim.initial_state,
        time_step_calculator=sim.time_step_calculator,
        step_fn=sim.step_fn,
    )
    history = output.StateHistory(torax_outputs)
    self._check_profiles_vs_expected(
        core_profiles=history.core_profiles,
        t=history.times,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=self.rtol,
        atol=self.atol,
    )


if __name__ == '__main__':
  absltest.main()
