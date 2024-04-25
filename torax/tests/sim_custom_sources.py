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

from __future__ import annotations

import dataclasses

from absl.testing import absltest
import chex
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.runtime_params import config_slice_args
from torax.sources import default_sources
from torax.sources import electron_density_sources
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
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

    def custom_source_formula(
        dynamic_config_slice: config_slice.DynamicConfigSlice,
        dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
        geo: geometry.Geometry,
        unused_state: state_lib.CoreProfiles | None,
    ):
      # Combine the outputs.
      assert isinstance(
          dynamic_source_runtime_params, _CustomSourceDynamicRuntimeParams
      )
      ignored_default_kwargs = dict(
          mode=dynamic_source_runtime_params.mode,
          is_explicit=dynamic_source_runtime_params.is_explicit,
          formula=dynamic_source_runtime_params.formula,
      )
      puff_params = electron_density_sources.DynamicGasPuffRuntimeParams(
          puff_decay_length=dynamic_source_runtime_params.puff_decay_length,
          S_puff_tot=dynamic_source_runtime_params.S_puff_tot,
          **ignored_default_kwargs,
      )
      nbi_params = electron_density_sources.DynamicNBIParticleRuntimeParams(
          nbi_deposition_location=dynamic_source_runtime_params.nbi_deposition_location,
          nbi_particle_width=dynamic_source_runtime_params.nbi_particle_width,
          S_nbi_tot=dynamic_source_runtime_params.S_nbi_tot,
          **ignored_default_kwargs,
      )
      pellet_params = electron_density_sources.DynamicPelletRuntimeParams(
          pellet_deposition_location=dynamic_source_runtime_params.pellet_deposition_location,
          pellet_width=dynamic_source_runtime_params.pellet_width,
          S_pellet_tot=dynamic_source_runtime_params.S_pellet_tot,
          **ignored_default_kwargs,
      )
      # pylint: disable=protected-access
      return (
          electron_density_sources._calc_puff_source(
              dynamic_config_slice=dynamic_config_slice,
              dynamic_source_runtime_params=puff_params,
              geo=geo,
          )
          + electron_density_sources._calc_nbi_source(
              dynamic_config_slice=dynamic_config_slice,
              dynamic_source_runtime_params=nbi_params,
              geo=geo,
          )
          + electron_density_sources._calc_pellet_source(
              dynamic_config_slice=dynamic_config_slice,
              dynamic_source_runtime_params=pellet_params,
              geo=geo,
          )
      )
      # pylint: enable=protected-access

    # First instantiate the same default sources that test_particle_sources
    # constant starts with.
    source_models = default_sources.get_default_sources()
    source_models.j_bootstrap.runtime_params.bootstrap_mult = 1
    source_models.qei_source.runtime_params.Qei_mult = 1
    nbi_params = source_models.sources['nbi_particle_source'].runtime_params
    assert isinstance(
        nbi_params, electron_density_sources.NBIParticleRuntimeParams
    )
    nbi_params.S_nbi_tot = 0.0
    pellet_params = source_models.sources['pellet_source'].runtime_params
    assert isinstance(
        pellet_params, electron_density_sources.PelletRuntimeParams
    )
    pellet_params.S_pellet_tot = 2.0e22
    gas_puff_params = source_models.sources['gas_puff_source'].runtime_params
    assert isinstance(
        gas_puff_params, electron_density_sources.GasPuffRuntimeParams
    )
    gas_puff_params.S_puff_tot = 1.0e22
    # Turn off some sources.
    source_models.sources['fusion_heat_source'].runtime_params.mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models.sources['ohmic_heat_source'].runtime_params.mode = (
        runtime_params_lib.Mode.ZERO
    )

    # Add the custom source with the correct params, but keep it turned off to
    # start.
    source_models.add_source(
        source_name=custom_source_name,
        source=source.SingleProfileSource(
            supported_modes=(
                runtime_params_lib.Mode.ZERO,
                runtime_params_lib.Mode.FORMULA_BASED,
            ),
            affected_core_profiles=(source.AffectedCoreProfile.NE,),
            formula=custom_source_formula,
            runtime_params=_CustomSourceRuntimeParams(
                mode=runtime_params_lib.Mode.ZERO,
                puff_decay_length=gas_puff_params.puff_decay_length,
                S_puff_tot=gas_puff_params.S_puff_tot,
                nbi_particle_width=nbi_params.nbi_particle_width,
                nbi_deposition_location=nbi_params.nbi_deposition_location,
                S_nbi_tot=nbi_params.S_nbi_tot,
                pellet_width=pellet_params.pellet_width,
                pellet_deposition_location=pellet_params.pellet_deposition_location,
                S_pellet_tot=pellet_params.S_pellet_tot,
            ),
        ),
    )

    # Copy the test_particle_sources_constant config in here for clarity.
    # These are the common kwargs without any of the sources.
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
        solver=config_lib.SolverConfig(
            predictor_corrector=False,
        ),
    )

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(
        'test_particle_sources_constant.h5', _ALL_PROFILES
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
      # Turn off the other sources and turn on the custom one.
      nbi_params.mode = runtime_params_lib.Mode.ZERO
      pellet_params.mode = runtime_params_lib.Mode.ZERO
      gas_puff_params.mode = runtime_params_lib.Mode.ZERO
      source_models.sources[custom_source_name].runtime_params.mode = (
          runtime_params_lib.Mode.FORMULA_BASED
      )
      self._run_sim_and_check(
          test_particle_sources_constant_config, sim, ref_profiles, ref_time
      )

    with self.subTest('without_defaults_and_without_custom_source'):
      # Confirm that the custom source actual has an effect.
      source_models.sources[custom_source_name].runtime_params.mode = (
          runtime_params_lib.Mode.ZERO
      )
      with self.assertRaises(AssertionError):
        self._run_sim_and_check(
            test_particle_sources_constant_config, sim, ref_profiles, ref_time
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
                sources_getter=lambda: sim.source_models.runtime_params,
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


# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class _CustomSourceRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime params for the custom source defined in the test case above."""

  puff_decay_length: runtime_params_lib.TimeDependentField
  S_puff_tot: runtime_params_lib.TimeDependentField
  nbi_particle_width: runtime_params_lib.TimeDependentField
  nbi_deposition_location: runtime_params_lib.TimeDependentField
  S_nbi_tot: runtime_params_lib.TimeDependentField
  pellet_width: runtime_params_lib.TimeDependentField
  pellet_deposition_location: runtime_params_lib.TimeDependentField
  S_pellet_tot: runtime_params_lib.TimeDependentField

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> _CustomSourceDynamicRuntimeParams:
    return _CustomSourceDynamicRuntimeParams(
        **config_slice_args.get_init_kwargs(
            input_config=self,
            output_type=_CustomSourceDynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class _CustomSourceDynamicRuntimeParams(
    runtime_params_lib.DynamicRuntimeParams
):
  puff_decay_length: float
  S_puff_tot: float
  nbi_particle_width: float
  nbi_deposition_location: float
  S_nbi_tot: float
  pellet_width: float
  pellet_deposition_location: float
  S_pellet_tot: float


# pylint: enable=invalid-name

if __name__ == '__main__':
  absltest.main()
