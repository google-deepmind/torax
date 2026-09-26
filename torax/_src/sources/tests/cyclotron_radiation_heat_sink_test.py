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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import cyclotron_radiation_heat_sink
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class CyclotronRadiationHeatSinkTest(parameterized.TestCase):
  """Unit tests for CyclotronRadiationHeatSink."""

  @parameterized.product(
      alpha_expected=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
      beta=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
      profile_edge_value=[0.3, 1.0],
      profile_core_multiplier=[5.0, 10.0],
  )
  def test_alpha_closed_form(
      self,
      alpha_expected,
      beta,
      profile_edge_value,
      profile_core_multiplier,
  ):
    """Test _alpha_closed_form in cyclotron_radiation_heat_sink."""

    rho_norm = jnp.linspace(0.0, 1.0, 25)
    profile_data = (
        profile_core_multiplier * (1 - rho_norm**beta) ** alpha_expected
        + profile_edge_value
    )

    alpha_closed_form_jitted = jax.jit(
        cyclotron_radiation_heat_sink._alpha_closed_form
    )

    # Calculate alpha with closed form formula.
    alpha = alpha_closed_form_jitted(
        beta=beta,
        rho_norm=rho_norm,
        profile_data=profile_data,
        profile_edge_value=profile_edge_value,
    )

    # Check that alpha is as expected
    np.testing.assert_allclose(alpha, alpha_expected, atol=1e-3)

  @parameterized.product(
      alpha=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
      beta=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
      profile_edge_value=[0.3, 1.0],
      profile_core_multiplier=[5.0, 10.0],
      beta_scan_parameters=[(0.5, 8, 32), (0.4, 6, 16)],
  )
  def test_solve_alpha_t_beta_t_grid_search(
      self,
      alpha,
      beta,
      profile_edge_value,
      profile_core_multiplier,
      beta_scan_parameters,
  ):
    """Test _solve_alpha_t_beta_t_grid_search in cyclotron_radiation_heat_sink."""

    beta_trials = jnp.linspace(
        beta_scan_parameters[0],
        beta_scan_parameters[1],
        beta_scan_parameters[2],
    )
    beta_expected = beta_trials[jnp.argmin(jnp.abs(beta_trials - beta))]

    rho_norm = jnp.linspace(0.0, 1.0, 25)
    profile_data = (
        profile_core_multiplier * (1 - rho_norm**beta) ** alpha
        + profile_edge_value
    )

    solve_alpha_t_beta_t_grid_search_jitted = jax.jit(
        cyclotron_radiation_heat_sink._solve_alpha_t_beta_t_grid_search,
        static_argnames=["beta_scan_parameters"],
    )

    # Calculate alpha with closed form formula.
    _, beta_grid_search = solve_alpha_t_beta_t_grid_search_jitted(
        rho_norm=rho_norm,
        te_data=profile_data,
        beta_scan_parameters=beta_scan_parameters,
    )

    # Check that beta is as expected
    np.testing.assert_allclose(beta_grid_search, beta_expected, atol=1e-7)

  @parameterized.named_parameters(
      dict(
          testcase_name="hollow_te_with_cold_edge",
          te_profile={0.0: 1.0, 0.9: 4.0, 1.0: 0.1},
          ne_profile={0.0: 1.5e20, 1.0: 0.8e20},
      ),
      dict(
          testcase_name="inverted_ne_with_peaked_te",
          te_profile={0.0: 2.0, 1.0: 1.8},
          ne_profile={0.0: 0.1e20, 1.0: 4.0e20},
      ),
      dict(
          testcase_name="monotonically_inverted_te_and_ne",
          te_profile={0.0: 1.0, 1.0: 5.0},
          ne_profile={0.0: 0.5e20, 1.0: 4.0e20},
      ),
      dict(
          testcase_name="non_monotonic_te_dipping_below_edge",
          te_profile={0.0: 2.0, 0.5: 0.5, 1.0: 1.0},
          ne_profile={0.0: 1.5e20, 1.0: 0.8e20},
      ),
      dict(
          testcase_name="flat_te_and_ne",
          te_profile={0.0: 2.0, 1.0: 2.0},
          ne_profile={0.0: 1.5e20, 1.0: 1.5e20},
      ),
  )
  def test_cyclotron_radiation_albajar_with_inverted_profiles(
      self,
      te_profile,
      ne_profile,
  ):
    config = default_configs.get_default_config_dict()
    config["sources"] = {
        cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_NAME: {}
    }
    config["profile_conditions"] = {
        "T_e": {0.0: te_profile},
        "n_e": {0.0: ne_profile},
        "n_e_nbar_is_fGW": False,
        "normalize_n_e_to_nbar": False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    alpha_n = cyclotron_radiation_heat_sink._alpha_closed_form(
        beta=2.0,
        rho_norm=geo.rho_face_norm,
        profile_data=core_profiles.n_e.face_value() / 1e20,
        profile_edge_value=0.0,
    )
    alpha_t, _ = (
        cyclotron_radiation_heat_sink._solve_alpha_t_beta_t_grid_search(
            rho_norm=geo.rho_face_norm,
            te_data=core_profiles.T_e.face_value(),
            beta_scan_parameters=(0.5, 8.0, 32),
        )
    )
    self.assertGreaterEqual(float(alpha_n), 0.0)
    self.assertGreaterEqual(float(alpha_t), 0.0)
    q_cycl = cyclotron_radiation_heat_sink.cyclotron_radiation_albajar(
        runtime_params=runtime_params,
        geo=geo,
        source_name=cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_NAME,
        core_profiles=core_profiles,
        unused_calculated_source_profiles=None,
        unused_conductivity=None,
    )[0]
    self.assertFalse(np.any(np.isnan(q_cycl)))
    self.assertTrue(np.all(np.isfinite(q_cycl)))
    self.assertTrue(np.all(q_cycl <= 0.0))


if __name__ == "__main__":
  absltest.main()
