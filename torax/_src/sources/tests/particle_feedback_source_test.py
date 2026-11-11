# Copyright 2026 DeepMind Technologies Limited
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
import jax.numpy as jnp
import numpy as np
import pydantic
import torax
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.sources import config as sources_config
from torax._src.sources import formulas
from torax._src.sources import generic_particle_source
from torax._src.sources import particle_feedback_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config as torax_pydantic_config
from torax._src.torax_pydantic import torax_pydantic


class ParticleFeedbackSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for ParticleFeedbackSource."""

  source_config_class = particle_feedback_source.ParticleFeedbackSourceConfig
  source_name = generic_particle_source.GenericParticleSource.SOURCE_ID
  model_name = 'feedback'

  def _make_feedback_setup(self, **source_overrides):
    config = default_configs.get_default_config_dict()
    config['sources'] = {
        'generic_particle': {
            'model_name': 'feedback',
            **source_overrides,
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    core_profiles, _, geo = core_profile_helpers.make_initial_inputs(
        torax_config
    )
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    source = torax_config.sources.build_models().standard_sources[
        'generic_particle'
    ]
    assert isinstance(source, particle_feedback_source.ParticleFeedbackSource)
    return core_profiles, geo, provider, source

  def test_proportional_term(self):
    base_profiles, geo, _, _ = self._make_feedback_setup()
    initial_line_avg = float(
        math_utils.line_average(base_profiles.kinetic.n_e.value, geo)
    )
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=10.0,
        target_average_n_e=initial_line_avg + 1e19,
        S_feedforward=0.0,
    )
    runtime_params = provider(t=0.0)
    source_params = runtime_params.sources['generic_particle']
    assert isinstance(source_params, particle_feedback_source.RuntimeParams)
    chex.assert_trees_all_close(source_params.proportional_gain, 10.0)

    output, state = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    profile = output.particle_source_e
    assert profile is not None
    assert isinstance(state, particle_feedback_source.ParticleFeedbackState)
    total_particles = math_utils.volume_integration(profile, geo)
    chex.assert_trees_all_close(total_particles, 1e20, rtol=1e-5)

  def test_integral_accumulates_and_clamps(self):
    base_profiles, geo, _, _ = self._make_feedback_setup()
    initial_line_avg = float(
        math_utils.line_average(base_profiles.kinetic.n_e.value, geo)
    )
    # Constant error = 1e19 m^-3, integral_gain = 20.0 m^3/s^2.
    # Each dt = 0.5 s adds 20.0 * 1e19 * 0.5 = 1e20 particles/s up to 1.5e20.
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=0.0,
        integral_gain=20.0,
        min_integral=-0.5e20,
        max_integral=1.5e20,
        target_average_n_e=initial_line_avg + 1e19,
        S_feedforward=0.0,
    )
    params_t0 = provider(t=0.0)
    state = source.initial_state(params_t0)

    # Initial step at t=0.0: integral_term = 0.0.
    output_0, state = source.get_value(
        runtime_params=params_t0,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state,
    )
    assert output_0.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_0.particle_source_e, geo),
        0.0,
        atol=1e-5,
    )

    # Re-evaluating at same t=0.0 (dt=0.0) does not double-accumulate.
    output_0_repeat, state = source.get_value(
        runtime_params=params_t0,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state,
    )
    assert output_0_repeat.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_0_repeat.particle_source_e, geo),
        0.0,
        atol=1e-5,
    )

    # Step 1 at t=0.5: S_i = 1e20.
    output_1, state = source.get_value(
        runtime_params=provider(t=0.5),
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state,
    )
    assert output_1.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_1.particle_source_e, geo),
        1e20,
        rtol=1e-5,
    )

    # Step 2 at t=1.0: unclamped would be 2e20, clamped to max_integral=1.5e20.
    output_2, state = source.get_value(
        runtime_params=provider(t=1.0),
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state,
    )
    assert output_2.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_2.particle_source_e, geo),
        1.5e20,
        rtol=1e-5,
    )

  def test_derivative_raw_and_smoothed(self):
    base_profiles, geo, _, _ = self._make_feedback_setup()
    initial_line_avg = float(
        math_utils.line_average(base_profiles.kinetic.n_e.value, geo)
    )
    # Target ramps linearly from initial_line_avg + 1e19 at t=0 to +3e19 at t=1,
    # so error(0) = 1e19, error(0.5) = 2e19, de/dt = 2e19 m^-3/s.
    target_ramp = {
        0.0: initial_line_avg + 1e19,
        1.0: initial_line_avg + 3e19,
    }
    core_profiles, geo, provider_raw, source = self._make_feedback_setup(
        proportional_gain=0.0,
        derivative_gain=5.0,
        derivative_smoothing_time_scale=0.0,
        target_average_n_e=target_ramp,
        S_feedforward=0.0,
    )
    params_t0 = provider_raw(t=0.0)
    output_0, state = source.get_value(
        runtime_params=params_t0,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(params_t0),
    )
    # No derivative spike on initial evaluation.
    assert output_0.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_0.particle_source_e, geo),
        0.0,
        atol=1e-5,
    )

    output_1, _ = source.get_value(
        runtime_params=provider_raw(t=0.5),
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state,
    )
    # Raw S_d = 5.0 * (2e19 - 1e19) / 0.5 = 1e20.
    assert output_1.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_1.particle_source_e, geo),
        1e20,
        rtol=1e-5,
    )

    # With exponential smoothing tau = 0.5 s over dt = 0.5 s:
    # alpha = 1 - exp(-1), smoothed S_d = alpha * 1e20.
    _, _, provider_smooth, _ = self._make_feedback_setup(
        proportional_gain=0.0,
        derivative_gain=5.0,
        derivative_smoothing_time_scale=0.5,
        target_average_n_e=target_ramp,
        S_feedforward=0.0,
    )
    params_smooth_t0 = provider_smooth(t=0.0)
    _, state_smooth = source.get_value(
        runtime_params=params_smooth_t0,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(params_smooth_t0),
    )
    output_smooth_1, _ = source.get_value(
        runtime_params=provider_smooth(t=0.5),
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state_smooth,
    )
    expected_smooth = float((1.0 - jnp.exp(-1.0)) * 1e20)
    assert output_smooth_1.particle_source_e is not None
    chex.assert_trees_all_close(
        math_utils.volume_integration(output_smooth_1.particle_source_e, geo),
        expected_smooth,
        rtol=1e-5,
    )

  def test_exponential_profile_shape(self):
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=0.0,
        S_feedforward=2e20,
        profile={
            'profile_type': 'exponential',
            'width': 0.08,
        },
    )
    params = provider(t=0.0)
    output, _ = source.get_value(
        runtime_params=params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(params),
    )
    profile = output.particle_source_e
    assert profile is not None
    expected = formulas.exponential_profile(
        geo=geo, decay_start=1.0, width=0.08, total=2e20
    )
    np.testing.assert_allclose(profile, expected, rtol=1e-6)
    chex.assert_trees_all_close(
        math_utils.volume_integration(profile, geo), 2e20, rtol=1e-5
    )

  def test_gaussian_profile_shape(self):
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=0.0,
        S_feedforward=3e20,
        profile={
            'profile_type': 'gaussian',
            'center': 0.3,
            'width': 0.15,
        },
    )
    params = provider(t=0.0)
    output, _ = source.get_value(
        runtime_params=params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(params),
    )
    profile = output.particle_source_e
    assert profile is not None
    expected = formulas.gaussian_profile(
        geo=geo, center=0.3, width=0.15, total=3e20
    )
    np.testing.assert_allclose(profile, expected, rtol=1e-6)
    chex.assert_trees_all_close(
        math_utils.volume_integration(profile, geo), 3e20, rtol=1e-5
    )

  def test_prescribed_profile_shape_rescales_volume_integral(self):
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=0.0,
        S_feedforward=4e20,
        profile={
            'profile_type': 'prescribed',
            'values': {0.0: {0.0: 2.0, 1.0: 6.0}},
        },
    )
    params = provider(t=0.0)
    output, _ = source.get_value(
        runtime_params=params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(params),
    )
    profile = output.particle_source_e
    assert profile is not None
    raw_shape = 2.0 + 4.0 * geo.rho_norm
    expected = raw_shape * (
        4e20 / math_utils.volume_integration(raw_shape, geo)
    )
    np.testing.assert_allclose(profile, expected, rtol=1e-6)
    chex.assert_trees_all_close(
        math_utils.volume_integration(profile, geo), 4e20, rtol=1e-5
    )

  def test_prescribed_zero_shape_is_zero_and_grad_safe(self):
    core_profiles, geo, provider, source = self._make_feedback_setup(
        proportional_gain=0.0,
        S_feedforward=4e20,
        profile={
            'profile_type': 'prescribed',
            'values': {0.0: {0.0: 0.0, 1.0: 0.0}},
        },
    )
    params = provider(t=0.0)
    state0 = source.initial_state(params)
    output, _ = source.get_value(
        runtime_params=params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=state0,
    )
    assert output.particle_source_e is not None
    np.testing.assert_allclose(output.particle_source_e, 0.0, atol=1e-12)

    def loss_fn(values: jax.Array) -> jax.Array:
      profile_params = particle_feedback_source.PrescribedProfileRuntimeParams(
          values=values
      )
      # pylint: disable=protected-access
      out = particle_feedback_source._compute_radial_profile(
          profile_params, jnp.asarray(4e20), geo
      )
      # pylint: enable=protected-access
      return jnp.sum(out)

    grad = jax.grad(loss_fn)(jnp.zeros_like(geo.rho_norm))
    self.assertFalse(bool(jnp.any(jnp.isnan(grad))))

  def test_nested_profile_update_fields_does_not_recompile_jit(self):
    sources = sources_config.SourcesConfig.from_dict({
        'generic_particle': {
            'model_name': 'feedback',
            'profile': {
                'profile_type': 'gaussian',
                'center': 0.2,
                'width': 0.1,
            },
        }
    })
    grid = torax_pydantic.Grid1D(face_centers=np.linspace(0.0, 1.0, 5))
    torax_pydantic.set_grid(sources, grid)
    compile_count = 0

    @jax.jit
    def eval_width(cfg: sources_config.SourcesConfig) -> chex.Numeric:
      nonlocal compile_count
      compile_count += 1
      src = cfg.generic_particle
      assert isinstance(
          src, particle_feedback_source.ParticleFeedbackSourceConfig
      )
      params = src.build_runtime_params(t=0.0)
      assert isinstance(
          params.profile, particle_feedback_source.GaussianProfileRuntimeParams
      )
      return params.profile.width

    chex.assert_trees_all_close(eval_width(sources), 0.1)
    sources._update_fields(  # pylint: disable=protected-access
        {'generic_particle.profile.width': 0.25}
    )
    torax_pydantic.set_grid(sources, grid, mode='relaxed')
    chex.assert_trees_all_close(eval_width(sources), 0.25)
    self.assertEqual(compile_count, 1)

  def test_prescribed_profile_set_grid_via_from_dict(self):
    src = particle_feedback_source.ParticleFeedbackSourceConfig.from_dict({
        'model_name': 'feedback',
        'profile': {
            'profile_type': 'prescribed',
            'values': {0.0: {0.0: 1.0, 1.0: 3.0}},
        },
    })
    grid = torax_pydantic.Grid1D(face_centers=np.linspace(0.0, 1.0, 5))
    torax_pydantic.set_grid(src, grid)
    params = src.build_runtime_params(t=0.0)
    assert isinstance(
        params.profile, particle_feedback_source.PrescribedProfileRuntimeParams
    )
    self.assertEqual(params.profile.values.shape, (4,))

  def test_invalid_integral_bounds_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'min_integral must be less than or equal to max_integral',
    ):
      particle_feedback_source.ParticleFeedbackSourceConfig.from_dict({
          'model_name': 'feedback',
          'min_integral': 1e22,
          'max_integral': -1e22,
      })

  @parameterized.product(
      is_explicit=(False, True),
      profile_cfg=(
          {'profile_type': 'exponential', 'width': 0.05},
          {'profile_type': 'gaussian', 'center': 0.0, 'width': 0.25},
          {
              'profile_type': 'prescribed',
              'values': {0.0: {0.0: 1.0, 1.0: 2.0}},
          },
      ),
  )
  def test_run_simulation_with_pid(
      self, is_explicit: bool, profile_cfg: dict[str, object]
  ):
    config = default_configs.get_default_config_dict()
    config['numerics']['evolve_density'] = True
    config['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 0.2,
        'dt': 0.1,
    }
    config['sources'] = {
        'generic_particle': {
            'model_name': 'feedback',
            'is_explicit': is_explicit,
            'proportional_gain': 1e2,
            'integral_gain': 5e2,
            'derivative_gain': 1e1,
            'derivative_smoothing_time_scale': 0.05,
            'min_integral': -1e22,
            'max_integral': 1e22,
            'target_average_n_e': 1.2e20,
            'S_feedforward': 1e21,
            'profile': profile_cfg,
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    state_history = torax.run_simulation(torax_config, progress_bar=False)
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)
    for core_sources in state_history.core_sources:
      particle_source_e = core_sources.sources[
          'generic_particle'
      ].particle_source_e
      assert particle_source_e is not None
      self.assertFalse(bool(jnp.any(jnp.isnan(particle_source_e))))


if __name__ == '__main__':
  absltest.main()
