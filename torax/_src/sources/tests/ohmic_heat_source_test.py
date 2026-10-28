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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import circular_geometry
from torax._src.physics import psi_calculations
from torax._src.sources import ohmic_heat_source
from torax._src.sources import source_state as source_state_lib
from torax._src.sources.tests import test_lib
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic


class OhmicHeatSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for OhmicHeatSource."""

  source_config_class = ohmic_heat_source.OhmicHeatSourceConfig
  source_name = ohmic_heat_source.OhmicHeatSource.SOURCE_ID
  needs_source_models = True

  def test_raises_error_if_calculated_source_profiles_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource()
    source_config = self._source_config_class.from_dict({})
    face_centers = interpolated_param_2d.get_face_centers(4)
    torax_pydantic.set_grid(
        source_config,
        torax_pydantic.Grid1D(face_centers=face_centers),
    )
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        instance=True,
        sources={self._source_name: source_config.build_runtime_params(t=0.0)},
    )
    with self.assertRaisesRegex(
        ValueError,
        'calculated_source_profiles is a required argument for'
        ' OhmicHeatSource._get_model_value. This can occur if this source'
        ' is used in an explicit source.',
    ):
      source.get_value(
          runtime_params,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=None,
          conductivity=mock.ANY,
          source_state=source_state_lib.SourceState(),
      )

  def test_raises_error_if_conductivity_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource()
    source_config = self._source_config_class.from_dict({})
    face_centers = interpolated_param_2d.get_face_centers(4)
    torax_pydantic.set_grid(
        source_config,
        torax_pydantic.Grid1D(face_centers=face_centers),
    )
    runtime_params = mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        instance=True,
        sources={self._source_name: source_config.build_runtime_params(t=0.0)},
    )
    with self.assertRaisesRegex(
        ValueError,
        'conductivity is a required argument for'
        ' OhmicHeatSource._get_model_value. This can occur if this source'
        ' is used in an explicit source.',
    ):
      source.get_value(
          runtime_params,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=mock.ANY,
          conductivity=None,
          source_state=source_state_lib.SourceState(),
      )

  @parameterized.parameters(-2.0, 2.0)
  @mock.patch.object(
      ohmic_heat_source.psi_calculations, 'calculate_psidot_from_psi_sources'
  )
  def test_ohmic_power_with_prescribed_psidot(
      self, psidot_val, mock_calculate_psidot
  ):
    n_rho = 10
    prescribed_psidot = jnp.full((n_rho,), psidot_val)

    runtime_params = mock.Mock()
    runtime_params.numerics.evolve_current = False
    runtime_params.profile_conditions.psidot = prescribed_psidot

    geo = mock.Mock()
    geo.rho_norm = jnp.linspace(0.05, 0.95, n_rho)
    geo.Phi_b = jnp.asarray(4.0)
    geo.Phi_b_dot = jnp.asarray(0.0)
    geo.F = jnp.full((n_rho,), 15.0)
    geo.g3 = jnp.full((n_rho,), 0.04)
    geo.gm5 = jnp.full((n_rho,), 9.0)

    core_profiles = mock.Mock()
    core_profiles.psi.grad.return_value = jnp.ones(n_rho)
    calculated_source_profiles = mock.Mock()
    conductivity = mock.Mock()
    conductivity.sigma = jnp.full((n_rho,), 1e6)

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        conductivity=conductivity,
        source_state=source_state_lib.SourceState(),
    )
    (pohm,) = output.profiles

    mock_calculate_psidot.assert_not_called()
    expected_e_dot_b = geo.F * geo.g3 / (2.0 * jnp.pi) * prescribed_psidot
    expected_pohm = conductivity.sigma * expected_e_dot_b**2 / geo.gm5
    np.testing.assert_allclose(np.asarray(pohm), expected_pohm)

  @mock.patch.object(
      ohmic_heat_source.psi_calculations, 'calculate_psidot_from_psi_sources'
  )
  def test_ohmic_power_subtracts_phi_b_dot_grid_motion(
      self, mock_calculate_psidot
  ):
    n_rho = 4
    rho_norm = jnp.linspace(0.1, 0.9, n_rho)
    phi_b = jnp.asarray(5.0)
    phi_b_dot = jnp.asarray(2.0)
    psi_grad = jnp.full((n_rho,), 3.0)
    # Set psidot equal to the pure grid-motion term so (dpsi/dt)|_Phi == 0.
    mock_calculate_psidot.return_value = (
        rho_norm * phi_b_dot / (2.0 * phi_b) * psi_grad
    )

    runtime_params = mock.Mock()
    runtime_params.numerics.evolve_current = True

    geo = mock.Mock()
    geo.rho_norm = rho_norm
    geo.Phi_b = phi_b
    geo.Phi_b_dot = phi_b_dot
    geo.F = jnp.full((n_rho,), 10.0)
    geo.g3 = jnp.full((n_rho,), 0.05)
    geo.gm5 = jnp.full((n_rho,), 4.0)

    core_profiles = mock.Mock()
    core_profiles.psi.grad.return_value = psi_grad
    calculated_source_profiles = mock.Mock()
    calculated_source_profiles.total_psi_sources.return_value = jnp.zeros(n_rho)
    conductivity = mock.Mock()
    conductivity.sigma = jnp.full((n_rho,), 1e6)

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        conductivity=conductivity,
        source_state=source_state_lib.SourceState(),
    )
    (pohm,) = output.profiles

    np.testing.assert_allclose(np.asarray(pohm), 0.0, atol=1e-12)

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
    runtime_params = mock.Mock()
    runtime_params.numerics.evolve_current = True

    core_profiles = mock.Mock()
    core_profiles.psi = psi
    calculated_source_profiles = mock.Mock()
    calculated_source_profiles.total_psi_sources.return_value = zero_psi_sources
    conductivity = mock.Mock()
    conductivity.sigma = sigma

    output, _ = ohmic_heat_source.OhmicHeatSource()._get_model_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        conductivity=conductivity,
        source_state=source_state_lib.SourceState(),
    )
    (pohm,) = output.profiles
    psidot = psi_calculations.calculate_psidot_from_psi_sources(
        psi_sources=zero_psi_sources,
        sigma=sigma,
        psi=psi,
        geo=geo,
    )
    large_aspect_ratio_pohm = (
        j_toroidal_total * psidot / (2.0 * jnp.pi * geo.R_major_profile)
    )
    np.testing.assert_allclose(
        np.asarray(pohm), large_aspect_ratio_pohm, rtol=3e-4
    )

if __name__ == '__main__':
  absltest.main()
