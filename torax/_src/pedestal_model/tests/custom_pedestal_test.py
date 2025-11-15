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
"""Tests for custom_pedestal module."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.pedestal_model import custom_pedestal


class CustomPedestalTest(parameterized.TestCase):
  """Tests for CustomPedestalModel."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    # Create mock geometry
    self.geo = mock.Mock()
    self.geo.torax_mesh.nx = 25
    self.geo.rho_norm = jnp.linspace(0, 1, 25)
    self.geo.a_minor = 1.0
    self.geo.B0 = 2.0

    # Create mock runtime params
    self.runtime_params = mock.Mock()
    self.runtime_params.profile_conditions.Ip = 15e6  # 15 MA
    self.runtime_params.pedestal = custom_pedestal.RuntimeParams(
        set_pedestal=True,
        rho_norm_ped_top=0.91,
        n_e_ped_is_fGW=False,
    )

    # Create mock core profiles
    self.core_profiles = mock.Mock()

  def test_custom_pedestal_with_simple_functions(self):
    """Test custom pedestal model with simple constant functions."""

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      return 5.0  # 5 keV

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      return 4.5  # 4.5 keV

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      return 0.7e20  # 0.7e20 m^-3

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    self.assertAlmostEqual(float(result.T_i_ped), 5.0, places=5)
    self.assertAlmostEqual(float(result.T_e_ped), 4.5, places=5)
    self.assertAlmostEqual(float(result.n_e_ped), 0.7e20, places=5)
    self.assertAlmostEqual(float(result.rho_norm_ped_top), 0.91, places=5)
    # Check that the index is correct
    expected_idx = np.argmin(np.abs(self.geo.rho_norm - 0.91))
    self.assertEqual(int(result.rho_norm_ped_top_idx), expected_idx)

  def test_custom_pedestal_with_scaling_laws(self):
    """Test custom pedestal model with physics-based scaling laws."""

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      # Simple EPED-like scaling: T_e ~ Ip^0.2 * B^0.8
      Ip_MA = runtime_params.profile_conditions.Ip / 1e6
      B_T = geo.B0
      return 0.5 * (Ip_MA**0.2) * (B_T**0.8)

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      # T_i = 1.2 * T_e
      T_e = T_e_ped_fn(runtime_params, geo, core_profiles)
      return 1.2 * T_e

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      # Return as Greenwald fraction
      return 0.7  # 0.7 * nGW

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
    )

    # Update runtime params to use Greenwald fraction
    self.runtime_params.pedestal = custom_pedestal.RuntimeParams(
        set_pedestal=True,
        rho_norm_ped_top=0.91,
        n_e_ped_is_fGW=True,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    # Calculate expected values
    Ip_MA = 15.0
    B_T = 2.0
    expected_T_e = 0.5 * (Ip_MA**0.2) * (B_T**0.8)
    expected_T_i = 1.2 * expected_T_e

    # Calculate Greenwald density
    nGW = Ip_MA / (np.pi * 1.0**2) * 1e20
    expected_n_e = 0.7 * nGW

    self.assertAlmostEqual(float(result.T_e_ped), expected_T_e, places=5)
    self.assertAlmostEqual(float(result.T_i_ped), expected_T_i, places=5)
    self.assertAlmostEqual(float(result.n_e_ped), expected_n_e, places=5)

  def test_custom_pedestal_with_rho_norm_function(self):
    """Test custom pedestal model with dynamic rho_norm_ped_top."""

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      return 5.0

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      return 4.5

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      return 0.7e20

    def rho_norm_ped_top_fn(runtime_params, geo, core_profiles):
      # Dynamic pedestal location based on current
      Ip_MA = runtime_params.profile_conditions.Ip / 1e6
      # Simple model: higher current -> narrower pedestal
      return 0.95 - 0.01 * Ip_MA

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
        rho_norm_ped_top_fn=rho_norm_ped_top_fn,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    # With Ip = 15 MA, rho_norm_ped_top should be 0.95 - 0.01*15 = 0.80
    expected_rho = 0.95 - 0.01 * 15.0
    self.assertAlmostEqual(float(result.rho_norm_ped_top), expected_rho, places=5)
    expected_idx = np.argmin(np.abs(self.geo.rho_norm - expected_rho))
    self.assertEqual(int(result.rho_norm_ped_top_idx), expected_idx)

  def test_custom_pedestal_absolute_density_units(self):
    """Test that absolute density units work correctly."""

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      return 5.0

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      return 4.5

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      return 0.8e20  # Absolute units

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
    )

    # Set n_e_ped_is_fGW to False (default)
    self.runtime_params.pedestal = custom_pedestal.RuntimeParams(
        set_pedestal=True,
        rho_norm_ped_top=0.91,
        n_e_ped_is_fGW=False,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    # Should return the value directly without Greenwald scaling
    self.assertAlmostEqual(float(result.n_e_ped), 0.8e20, places=5)

  def test_custom_pedestal_greenwald_fraction(self):
    """Test that Greenwald fraction conversion works correctly."""

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      return 5.0

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      return 4.5

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      return 0.6  # 0.6 * nGW

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
    )

    # Set n_e_ped_is_fGW to True
    self.runtime_params.pedestal = custom_pedestal.RuntimeParams(
        set_pedestal=True,
        rho_norm_ped_top=0.91,
        n_e_ped_is_fGW=True,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    # Calculate expected Greenwald density
    Ip_MA = 15.0
    a_minor = 1.0
    nGW = Ip_MA / (np.pi * a_minor**2) * 1e20
    expected_n_e = 0.6 * nGW

    self.assertAlmostEqual(float(result.n_e_ped), expected_n_e, places=5)

  def test_custom_pedestal_uses_geometry_parameters(self):
    """Test that custom functions can access geometry parameters."""

    def T_e_ped_fn(runtime_params, geo, core_profiles):
      # Scale with magnetic field
      return 2.0 * geo.B0

    def T_i_ped_fn(runtime_params, geo, core_profiles):
      # Scale with aspect ratio
      return 3.0 * geo.a_minor

    def n_e_ped_fn(runtime_params, geo, core_profiles):
      return 0.7e20

    model = custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=T_i_ped_fn,
        T_e_ped_fn=T_e_ped_fn,
        n_e_ped_fn=n_e_ped_fn,
    )

    result = model(self.runtime_params, self.geo, self.core_profiles)

    # T_e should be 2.0 * B0 = 2.0 * 2.0 = 4.0
    self.assertAlmostEqual(float(result.T_e_ped), 4.0, places=5)
    # T_i should be 3.0 * a_minor = 3.0 * 1.0 = 3.0
    self.assertAlmostEqual(float(result.T_i_ped), 3.0, places=5)


if __name__ == '__main__':
  absltest.main()
