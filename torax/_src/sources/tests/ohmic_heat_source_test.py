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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.core_profiles import initialization
from torax._src.geometry import circular_geometry
from torax._src.physics import psi_calculations
from torax._src.sources import ohmic_heat_source
from torax._src.sources import source_state as source_state_lib
from torax._src.sources.tests import test_lib


class OhmicHeatSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for OhmicHeatSource."""

  source_config_class = ohmic_heat_source.OhmicHeatSourceConfig
  source_name = ohmic_heat_source.OhmicHeatSource.SOURCE_ID
  needs_source_models = True
  can_be_explicit = False

  @parameterized.parameters(-2.0, 2.0)
  def test_ohmic_power_with_psidot(self, psidot_val):
    n_rho = 10
    psidot = jnp.full((n_rho,), psidot_val)

    geo = mock.Mock()
    geo.rho_norm = jnp.linspace(0.05, 0.95, n_rho)
    geo.Phi_b = jnp.asarray(4.0)
    geo.Phi_b_dot = jnp.asarray(0.0)
    geo.F = jnp.full((n_rho,), 15.0)
    geo.g3 = jnp.full((n_rho,), 0.04)
    geo.gm5 = jnp.full((n_rho,), 9.0)

    core_profiles = mock.Mock()
    core_profiles.poloidal_flux.psi.grad.return_value = jnp.ones(n_rho)
    core_profiles.psidot = mock.Mock(value=psidot)
    conductivity = mock.Mock()
    conductivity.sigma = jnp.full((n_rho,), 1e6)
    core_profiles.conductivity = conductivity

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=mock.ANY,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source_state_lib.SourceState(),
    )
    assert output.power_density_e is not None

    expected_e_dot_b = geo.F * geo.g3 / (2.0 * jnp.pi) * psidot
    expected_pohm = conductivity.sigma * expected_e_dot_b**2 / geo.gm5
    np.testing.assert_allclose(
        np.asarray(output.power_density_e), expected_pohm
    )

  def test_ohmic_power_subtracts_phi_b_dot_grid_motion(self):
    n_rho = 4
    rho_norm = jnp.linspace(0.1, 0.9, n_rho)
    phi_b = jnp.asarray(5.0)
    phi_b_dot = jnp.asarray(2.0)
    psi_grad = jnp.full((n_rho,), 3.0)
    # Set psidot equal to the pure grid-motion term so (dpsi/dt)|_Phi == 0.
    psidot = rho_norm * phi_b_dot / (2.0 * phi_b) * psi_grad

    geo = mock.Mock()
    geo.rho_norm = rho_norm
    geo.Phi_b = phi_b
    geo.Phi_b_dot = phi_b_dot
    geo.F = jnp.full((n_rho,), 10.0)
    geo.g3 = jnp.full((n_rho,), 0.05)
    geo.gm5 = jnp.full((n_rho,), 4.0)

    core_profiles = mock.Mock()
    core_profiles.poloidal_flux.psi.grad.return_value = psi_grad
    core_profiles.psidot = mock.Mock(value=psidot)
    conductivity = mock.Mock()
    conductivity.sigma = jnp.full((n_rho,), 1e6)
    core_profiles.conductivity = conductivity

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=mock.ANY,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source_state_lib.SourceState(),
    )
    assert output.power_density_e is not None

    np.testing.assert_allclose(
        np.asarray(output.power_density_e), 0.0, atol=1e-12
    )

  def test_ohmic_power_agrees_with_large_aspect_ratio_formula_for_pure_ohmic_current(
      self,
  ):
    geo = circular_geometry.CircularConfig(
        n_rho=25,
        elongation_LCFS=1.0,
        R_major=100.0,
        a_minor=1.0,
        B_0=5.0,
    ).build_geometry()
    ip_tot = 1e6
    j_toroidal_total_hires_shape = (1.0 - geo.rho_hires_norm**2) ** 2
    j_toroidal_total_hires = jnp.asarray(
        j_toroidal_total_hires_shape
        * (
            ip_tot
            / jnp.trapezoid(
                j_toroidal_total_hires_shape * geo.spr_hires, geo.rho_hires_norm
            )
        )
    )
    psi = initialization.update_psi_from_j(ip_tot, geo, j_toroidal_total_hires)
    j_toroidal_total, _, _ = psi_calculations.calc_j_toroidal_total(
        geo, psi, min_rho_norm=0.01
    )

    sigma = jnp.full_like(geo.rho_norm, 1e6)
    zero_psi_sources = jnp.zeros_like(geo.rho_norm)
    psidot = psi_calculations.calculate_psidot(
        psi=psi,
        sigma=sigma,
        psi_sources=zero_psi_sources,
        geo=geo,
        evolve_current=True,
        right_face_grad_constraint=jnp.zeros((), dtype=jax_utils.get_dtype()),
    )

    core_profiles = mock.Mock()
    core_profiles.poloidal_flux.psi = psi
    core_profiles.psidot = psidot
    conductivity = mock.Mock()
    conductivity.sigma = sigma
    core_profiles.conductivity = conductivity

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=mock.ANY,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source_state_lib.SourceState(),
    )
    assert output.power_density_e is not None
    large_aspect_ratio_pohm = (
        j_toroidal_total * psidot.value / (2.0 * jnp.pi * geo.R_major_profile)
    )
    np.testing.assert_allclose(
        np.asarray(output.power_density_e), large_aspect_ratio_pohm, rtol=3e-4
    )

if __name__ == '__main__':
  absltest.main()
