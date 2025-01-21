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
from torax import array_typing
from torax import interpolated_param
from torax import output
from torax import sim as sim_lib
from torax import state as state_lib
from torax.config import config_args
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider
from torax.pedestal_model import set_tped_nped
from torax.sources import electron_density_sources
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources.tests import test_lib
from torax.stepper import linear_theta_method
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import sim_test_case
from torax.transport_model import constant as constant_transport_model


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimWithCustomSourcesTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim with custom sources."""

  def setUp(self):
    super().setUp()
    self.source_models_builder = default_sources.get_default_sources_builder()
    pedestal_runtime_params = set_tped_nped.RuntimeParams()
    self.basic_pedestal_model_builder = (
        set_tped_nped.SetTemperatureDensityPedestalModelBuilder(
            runtime_params=pedestal_runtime_params
        )
    )
    self.constant_transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                De_const=0.5,
                Ve_const=-0.2,
            ),
        )
    )
    self.stepper_builder = linear_theta_method.LinearThetaMethodBuilder(
        runtime_params=linear_theta_method.LinearRuntimeParams(
            predictor_corrector=False,
        )
    )
    # Copy the test_particle_sources_constant config in here for clarity.
    # These are the common kwargs without any of the sources.
    self.test_particle_sources_constant_runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=True,
            nbar=0.85,
            nu=0,
            ne_bound_right=0.5,
        ),
        numerics=numerics_lib.Numerics(
            ion_heat_eq=True,
            el_heat_eq=True,
            dens_eq=True,  # This is important to be True to test ne sources.
            current_eq=True,
            resistivity_mult=100,
            t_final=2,
        ),
    )

  def test_custom_ne_source_can_replace_defaults(self):
    """Replaces all the default ne sources with a custom one."""

    # For this example, use test_particle_sources_constant with the linear
    # stepper.
    custom_source_name = 'foo'

    def custom_source_formula(
        static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
        dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
        geo: geometry.Geometry,
        unused_source_name: str,
        unused_state: state_lib.CoreProfiles | None,
        unused_source_models: ...,
    ):
      # Combine the outputs.
      # pylint: disable=protected-access
      return (
          electron_density_sources.calc_puff_source(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              static_runtime_params_slice=static_runtime_params_slice,
              geo=geo,
              source_name=electron_density_sources.GasPuffSource.SOURCE_NAME,
          )
          + electron_density_sources.calc_generic_particle_source(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              static_runtime_params_slice=static_runtime_params_slice,
              geo=geo,
              source_name=electron_density_sources.GenericParticleSource.SOURCE_NAME,
          )
          + electron_density_sources.calc_pellet_source(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              static_runtime_params_slice=static_runtime_params_slice,
              geo=geo,
              source_name=electron_density_sources.PelletSource.SOURCE_NAME,
          )
      )
      # pylint: enable=protected-access

    # First instantiate the same default sources that test_particle_sources
    # constant starts with.
    source_models_builder = self.source_models_builder
    source_models_builder.runtime_params['j_bootstrap'].bootstrap_mult = 1
    source_models_builder.runtime_params['qei_source'].Qei_mult = 1
    params = source_models_builder.runtime_params['generic_particle_source']
    assert isinstance(
        params, electron_density_sources.GenericParticleSourceRuntimeParams
    )
    params.S_tot = 0.0
    pellet_params = source_models_builder.runtime_params['pellet_source']
    assert isinstance(
        pellet_params, electron_density_sources.PelletRuntimeParams
    )
    pellet_params.S_pellet_tot = 2.0e22
    gas_puff_params = source_models_builder.runtime_params['gas_puff_source']
    assert isinstance(
        gas_puff_params, electron_density_sources.GasPuffRuntimeParams
    )
    gas_puff_params.S_puff_tot = 1.0e22
    # Turn off some sources.
    source_models_builder.runtime_params['fusion_heat_source'].mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models_builder.runtime_params['ohmic_heat_source'].mode = (
        runtime_params_lib.Mode.ZERO
    )
    source_models_builder.runtime_params['bremsstrahlung_heat_sink'].mode = (
        runtime_params_lib.Mode.ZERO
    )

    # Add the custom source with the correct params, but keep it turned off to
    # start.
    source_builder = source_lib.make_source_builder(
        test_lib.TestSource,
        runtime_params_type=_CustomSourceRuntimeParams,
        model_func=custom_source_formula,
    )
    runtime_params = _CustomSourceRuntimeParams(
        mode=runtime_params_lib.Mode.ZERO,
        puff_decay_length=gas_puff_params.puff_decay_length,
        S_puff_tot=gas_puff_params.S_puff_tot,
        particle_width=params.particle_width,
        deposition_location=params.deposition_location,
        S_tot=params.S_tot,
        pellet_width=pellet_params.pellet_width,
        pellet_deposition_location=pellet_params.pellet_deposition_location,
        S_pellet_tot=pellet_params.S_pellet_tot,
    )
    source_models_builder.source_builders[custom_source_name] = (
        source_builder(
            runtime_params=runtime_params,
        )
    )

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(
        'test_particle_sources_constant.nc', _ALL_PROFILES
    )
    geo_provider = geometry_provider.ConstantGeometryProvider(
        geometry.build_circular_geometry()
    )
    sim = sim_lib.Sim.create(
        runtime_params=self.test_particle_sources_constant_runtime_params,
        geometry_provider=geo_provider,
        stepper_builder=self.stepper_builder,
        transport_model_builder=self.constant_transport_model_builder,
        source_models_builder=source_models_builder,
        pedestal_model_builder=self.basic_pedestal_model_builder,
    )

    # Make sure the config copied here works with these references.
    with self.subTest('with_defaults_and_without_custom_source'):
      # Need to run the sim once to build the step_fn.
      sim_outputs = sim.run()
      history = output.StateHistory(sim_outputs, sim.source_models)
      self._check_profiles_vs_expected(
          core_profiles=history.core_profiles,
          t=history.times,
          ref_time=ref_time,
          ref_profiles=ref_profiles,
          rtol=self.rtol,
          atol=self.atol,
      )

    with self.subTest('without_defaults_and_with_custom_source'):
      # Turn off the other sources and turn on the custom one.
      params.mode = runtime_params_lib.Mode.ZERO
      pellet_params.mode = runtime_params_lib.Mode.ZERO
      gas_puff_params.mode = runtime_params_lib.Mode.ZERO
      runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
      self._run_sim_and_check(sim, ref_profiles, ref_time)

    with self.subTest('without_defaults_and_without_custom_source'):
      # Confirm that the custom source actual has an effect.
      runtime_params.mode = runtime_params_lib.Mode.ZERO
      with self.assertRaises(AssertionError):
        self._run_sim_and_check(sim, ref_profiles, ref_time)

  def _run_sim_and_check(
      self,
      sim: sim_lib.Sim,
      ref_profiles: dict[str, chex.ArrayTree],
      ref_time: chex.Array,
  ):
    """Runs sim with new runtime params and checks the profiles vs. expected."""
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params=self.test_particle_sources_constant_runtime_params,
            source_runtime_params=self.source_models_builder.runtime_params,
            torax_mesh=sim.geometry_provider.torax_mesh,
            stepper=self.stepper_builder.runtime_params,
        )
    )
    sim._static_runtime_params_slice = static_runtime_params_slice  # pylint: disable=protected-access
    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs, sim.source_models)
    self._check_profiles_vs_expected(
        core_profiles=history.core_profiles,
        t=history.times,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=self.rtol,
        atol=self.atol,
    )


# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class _CustomSourceRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime params for the custom source defined in the test case above."""

  puff_decay_length: runtime_params_lib.TimeInterpolatedInput
  S_puff_tot: runtime_params_lib.TimeInterpolatedInput
  particle_width: runtime_params_lib.TimeInterpolatedInput
  deposition_location: runtime_params_lib.TimeInterpolatedInput
  S_tot: runtime_params_lib.TimeInterpolatedInput
  pellet_width: runtime_params_lib.TimeInterpolatedInput
  pellet_deposition_location: runtime_params_lib.TimeInterpolatedInput
  S_pellet_tot: runtime_params_lib.TimeInterpolatedInput

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> '_CustomSourceRuntimeParamsProvider':
    if torax_mesh is None:
      raise ValueError('torax_mesh is required for CustomSourceRuntimeParams.')
    return _CustomSourceRuntimeParamsProvider(
        runtime_params_config=self,
        prescribed_values=config_args.get_interpolated_var_2d(
            self.prescribed_values, torax_mesh.cell_centers
        ),
        puff_decay_length=config_args.get_interpolated_var_single_axis(
            self.puff_decay_length
        ),
        S_puff_tot=config_args.get_interpolated_var_single_axis(
            self.S_puff_tot
        ),
        particle_width=config_args.get_interpolated_var_single_axis(
            self.particle_width
        ),
        deposition_location=config_args.get_interpolated_var_single_axis(
            self.deposition_location
        ),
        S_tot=config_args.get_interpolated_var_single_axis(self.S_tot),
        pellet_width=config_args.get_interpolated_var_single_axis(
            self.pellet_width
        ),
        pellet_deposition_location=config_args.get_interpolated_var_single_axis(
            self.pellet_deposition_location
        ),
        S_pellet_tot=config_args.get_interpolated_var_single_axis(
            self.S_pellet_tot
        ),
    )


@chex.dataclass
class _CustomSourceRuntimeParamsProvider(
    runtime_params_lib.RuntimeParamsProvider
):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: _CustomSourceRuntimeParams
  puff_decay_length: interpolated_param.InterpolatedVarSingleAxis
  S_puff_tot: interpolated_param.InterpolatedVarSingleAxis
  particle_width: interpolated_param.InterpolatedVarSingleAxis
  deposition_location: interpolated_param.InterpolatedVarSingleAxis
  S_tot: interpolated_param.InterpolatedVarSingleAxis
  pellet_width: interpolated_param.InterpolatedVarSingleAxis
  pellet_deposition_location: interpolated_param.InterpolatedVarSingleAxis
  S_pellet_tot: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> '_CustomSourceDynamicRuntimeParams':
    return _CustomSourceDynamicRuntimeParams(
        **self.get_dynamic_params_kwargs(t)
    )


@chex.dataclass(frozen=True)
class _CustomSourceDynamicRuntimeParams(
    runtime_params_lib.DynamicRuntimeParams
):
  puff_decay_length: array_typing.ScalarFloat
  S_puff_tot: array_typing.ScalarFloat
  particle_width: array_typing.ScalarFloat
  deposition_location: array_typing.ScalarFloat
  S_tot: array_typing.ScalarFloat
  pellet_width: array_typing.ScalarFloat
  pellet_deposition_location: array_typing.ScalarFloat
  S_pellet_tot: array_typing.ScalarFloat


# pylint: enable=invalid-name

if __name__ == '__main__':
  absltest.main()
