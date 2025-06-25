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
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from torax._src.sources import cyclotron_radiation_heat_sink


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


if __name__ == "__main__":
  absltest.main()
