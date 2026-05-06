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
"""Tests for the scaled_profile ICRH model."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.sources.ion_cyclotron_source import scaled_profile
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class ScaledProfileIonCyclotronSourceTest(test_lib.SourceTestCase):

  source_name = icrh_base.IonCyclotronSource.SOURCE_NAME
  source_config_class = scaled_profile.ScaledProfileIonCyclotronSourceConfig

  def _build_and_run(self, icrh_config_dict, n_rho=None):
    """Helper to build a ToraxConfig and run the ICRH source."""
    config = default_configs.get_default_config_dict()
    if n_rho is not None:
      config['geometry']['n_rho'] = n_rho
    config['sources'] = {
        icrh_base.IonCyclotronSource.SOURCE_NAME: icrh_config_dict,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    icrh_source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = icrh_source.get_value(
        runtime_params,
        geo,
        core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    return output, geo

  @parameterized.parameters(5e6, 10e6, 25e6)
  def test_scaled_profile_matches_total_power(self, total_power):
    """Volume-integrated heating should equal P_total * absorption_fraction."""
    # Create a peaked Gaussian-like reference profile.
    rho = np.linspace(0, 1, 25)
    profile = np.exp(-((rho - 0.3) ** 2) / (2 * 0.1**2))

    config_dict = {
        'model_name': 'scaled_profile',
        'P_total': total_power,
        'absorption_fraction': 0.9,
        'heat_profile_ion': (rho, profile),
        'heat_profile_electron': (rho, profile),
        'reference_B0': 5.3,  # Matches default circular geometry B_0.
    }
    output, geo = self._build_and_run(config_dict)
    ion_el_total = output[0] + output[1]
    integrated_power = jnp.sum(ion_el_total * geo.vpr * geo.drho_norm)
    np.testing.assert_allclose(
        integrated_power,
        total_power * config_dict['absorption_fraction'],
        rtol=1e-4,
    )

  def test_scaled_profile_no_shift_at_reference_B0(self):
    """When B0 == reference_B0, profiles should be proportional to reference."""
    rho = np.linspace(0, 1, 25)
    # Use broad profiles that are well-resolved on the grid.
    ion_profile = np.exp(-((rho - 0.4) ** 2) / (2 * 0.2**2))
    el_profile = np.exp(-((rho - 0.5) ** 2) / (2 * 0.2**2))

    output, geo = self._build_and_run(
        {
            'model_name': 'scaled_profile',
            'P_total': 10e6,
            'absorption_fraction': 1.0,
            'heat_profile_ion': (rho, ion_profile),
            'heat_profile_electron': (rho, el_profile),
            'reference_B0': 5.3,  # Matches default circular geo B_0.
        },
        n_rho=25,
    )

    # Since B0 == reference_B0, no shift occurs — only uniform amplitude
    # scaling. The output profiles should be proportional to the interpolated
    # reference profiles (i.e. out = constant * ref everywhere).
    out_ion = np.array(output[0])
    out_el = np.array(output[1])
    rho_grid = np.array(geo.torax_mesh.cell_centers)
    ref_ion_interp = np.interp(rho_grid, rho, ion_profile)
    ref_el_interp = np.interp(rho_grid, rho, el_profile)
    # Check proportionality where profiles are non-negligible.
    ref_total_interp = ref_ion_interp + ref_el_interp
    mask = ref_total_interp > np.max(ref_total_interp) * 1e-3
    scale_ion = out_ion[mask] / ref_ion_interp[mask]
    scale_el = out_el[mask] / ref_el_interp[mask]
    # All scale factors should be the same constant.
    np.testing.assert_allclose(scale_ion, scale_ion[0], rtol=1e-5)
    np.testing.assert_allclose(scale_el, scale_el[0], rtol=1e-5)
    np.testing.assert_allclose(scale_ion[0], scale_el[0], rtol=1e-5)

  def test_scaled_profile_returns_zero_fast_ions(self):
    """The scaled_profile model should always return zero fast ions."""
    rho = np.linspace(0, 1, 25)
    profile = np.exp(-((rho - 0.3) ** 2) / (2 * 0.1**2))

    output, _ = self._build_and_run({
        'model_name': 'scaled_profile',
        'P_total': 10e6,
        'absorption_fraction': 1.0,
        'heat_profile_ion': (rho, profile),
        'heat_profile_electron': (rho, profile),
        'reference_B0': 12.2,
    })
    fast_ions = output[2]
    for fi in fast_ions:
      np.testing.assert_allclose(fi.n.value, 0.0, atol=1e-15)

  @parameterized.parameters(4.8, 5.8)
  def test_b0_shift_moves_peak(self, reference_B0):
    """Profile peak should shift to the analytically expected location."""
    rho_peak_ref = 0.5
    n_rho = 50
    rho = np.linspace(0, 1, n_rho)
    profile = np.exp(-((rho - rho_peak_ref) ** 2) / (2 * 0.1**2))

    output, geo = self._build_and_run(
        {
            'model_name': 'scaled_profile',
            'P_total': 10e6,
            'absorption_fraction': 1.0,
            'heat_profile_ion': (rho, profile),
            'heat_profile_electron': (rho, np.zeros_like(profile)),
            'reference_B0': reference_B0,
        },
        n_rho=n_rho,
    )

    # For circular geometry: R_out(ρ) = R_major + ρ * r_minor.
    # The model shifts R → R * B_ratio and maps back to ρ, so the
    # output peak appears at:
    #   ρ_out = ((R_major + ρ_peak * r_minor) / B_ratio - R_major) / r_minor
    B_ratio = geo.B_0 / reference_B0
    expected_rho = (
        (geo.R_major + rho_peak_ref * geo.a_minor) / B_ratio - geo.R_major
    ) / geo.a_minor

    rho_grid = np.array(geo.torax_mesh.cell_centers)
    expected_index = int(np.argmin(np.abs(rho_grid - expected_rho)))
    actual_index = int(np.argmax(np.array(output[0])))
    self.assertEqual(actual_index, expected_index)


if __name__ == '__main__':
  absltest.main()
