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

"""Tests for using custom, user-defined sources/sinks within TORAX."""

from absl.testing import absltest
import chex
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.sources import electron_density_sources
from torax.sources import source
from torax.sources import source_config
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.tests.test_lib import sim_test_case
from torax.transport_model import constant as constant_transport_model


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimWithCustomSourcesTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim with custom sources."""

  def test_custom_ne_source_can_replace_defaults(self):
    """Replaces all the default ne sources with a custom one."""

    # For this example, use test_particle_sources_constant with the linear
    # stepper.
    custom_source_name = 'custom_ne_source'

    def custom_source_formula(dynamic_config, geo, unused_state):
      # Combine the outputs of the pellet
      return (
          electron_density_sources.calc_puff_source(
              geo,
              puff_decay_length=dynamic_config.puff_decay_length,
              S_puff_tot=dynamic_config.S_puff_tot,
              nref=dynamic_config.nref,
          )
          + electron_density_sources.calc_nbi_source(
              geo,
              nbi_deposition_location=dynamic_config.nbi_deposition_location,
              nbi_particle_width=dynamic_config.nbi_particle_width,
              S_nbi_tot=dynamic_config.S_nbi_tot,
              nref=dynamic_config.nref,
          )
          + electron_density_sources.calc_pellet_source(
              geo,
              pellet_deposition_location=(
                  dynamic_config.pellet_deposition_location
              ),
              pellet_width=dynamic_config.pellet_width,
              S_pellet_tot=dynamic_config.S_pellet_tot,
              nref=dynamic_config.nref,
          )
      )

    source_models = source_models_lib.SourceModels(
        additional_sources=[
            source.SingleProfileSource(
                name=custom_source_name,
                supported_types=(
                    source_config.SourceType.ZERO,
                    source_config.SourceType.FORMULA_BASED,
                ),
                affected_core_profiles=(source.AffectedCoreProfile.NE,),
                formula=custom_source_formula,
            )
        ]
    )

    # Copy the test_particle_sources_constant config in here for clarity.
    # These are the common kwargs without any of the sources.
    test_particle_sources_constant_config_kwargs = dict(
        profile_conditions=config_lib.ProfileConditions(
            set_pedestal=True,
            nbar=0.85,
        ),
        numerics=config_lib.Numerics(
            Qei_mult=1,
            ion_heat_eq=True,
            el_heat_eq=True,
            dens_eq=True,  # This is important to be True to test ne sources.
            current_eq=True,
            resistivity_mult=100,
            bootstrap_mult=1,
            t_final=2,
        ),
        nu=0,
        S_pellet_tot=2.0e22,
        S_puff_tot=1.0e22,
        S_nbi_tot=0.0,
        solver=config_lib.SolverConfig(
            predictor_corrector=False,
        ),
    )
    # We need to turn off some other sources for test_particle_sources_constant
    # that are unrelated to our test for the ne custom source.
    unrelated_source_configs = dict(
        fusion_heat_source=source_config.SourceConfig(
            source_type=source_config.SourceType.ZERO,
        ),
        ohmic_heat_source=source_config.SourceConfig(
            source_type=source_config.SourceType.ZERO,
        ),
    )

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(
        'test_particle_sources_constant.h5', _ALL_PROFILES
    )

    # Set up the sim with the original config. We set up the sim only once and
    # update the config on each run below in a way that does not trigger
    # recompiles. This way we only trace the code once.
    test_particle_sources_constant_config = config_lib.Config(
        **test_particle_sources_constant_config_kwargs,
        sources=dict(
            **unrelated_source_configs,
            # Turn off the custom source
            custom_ne_source=source_config.SourceConfig(
                source_type=source_config.SourceType.ZERO,
            ),
        ),
    )
    geo = geometry.build_circular_geometry(
        test_particle_sources_constant_config
    )
    sim = sim_lib.build_sim_from_config(
        config=test_particle_sources_constant_config,
        geo=geo,
        stepper_builder=linear_theta_method.LinearThetaMethod,
        transport_model=constant_transport_model.ConstantTransportModel(
            runtime_params=constant_transport_model.RuntimeParams(
                De_const=0.5,
                Ve_const=-0.2,
            ),
        ),
        source_models=source_models,
    )

    # Make sure the config copied here works with these references.
    with self.subTest('with_defaults_and_without_custom_source'):
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

    with self.subTest('without_defaults_and_with_custom_source'):
      config_with_custom_source = config_lib.Config(
          **test_particle_sources_constant_config_kwargs,
          sources=dict(
              **unrelated_source_configs,
              custom_ne_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.FORMULA_BASED,
              ),
              gas_puff_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
              nbi_particle_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
              pellet_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
          ),
      )
      self._run_sim_and_check(
          config_with_custom_source, sim, ref_profiles, ref_time
      )

    with self.subTest('without_defaults_and_without_custom_source'):
      # Confirm that the custom source actual has an effect.
      config_without_ne_sources = config_lib.Config(
          **test_particle_sources_constant_config_kwargs,
          sources=dict(
              **unrelated_source_configs,
              custom_ne_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
              gas_puff_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
              nbi_particle_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
              pellet_source=source_config.SourceConfig(
                  source_type=source_config.SourceType.ZERO,
              ),
          ),
      )
      with self.assertRaises(AssertionError):
        self._run_sim_and_check(
            config_without_ne_sources, sim, ref_profiles, ref_time
        )

  def _run_sim_and_check(
      self,
      config: config_lib.Config,
      sim: sim_lib.Sim,
      ref_profiles: dict[str, chex.ArrayTree],
      ref_time: chex.Array,
  ):
    """Runs sim with new dynamic config and checks the profiles vs. expected."""
    torax_outputs = sim_lib.run_simulation(
        initial_state=sim.initial_state,
        step_fn=sim.step_fn,
        geometry_provider=sim.geometry_provider,
        dynamic_config_slice_provider=(
            config_slice.DynamicConfigSliceProvider(
                config=config,
                transport_getter=lambda: sim.transport_model.runtime_params,
            )
        ),
        static_config_slice=sim.static_config_slice,
        time_step_calculator=sim.time_step_calculator,
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
