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
import jax.numpy as jnp
import numpy as np
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.sources import ohmic_heat_source
from torax._src.sources.tests import test_lib
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic


class OhmicHeatSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for OhmicHeatSource."""

  def setUp(self):
    super().setUp(
        source_config_class=ohmic_heat_source.OhmicHeatSourceConfig,
        source_name=ohmic_heat_source.OhmicHeatSource.SOURCE_NAME,
        needs_source_models=True,
    )

  def test_raises_error_if_calculated_source_profiles_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource(
        model_func=ohmic_heat_source.ohmic_model_func
    )
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
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.',
    ):
      source.get_value(
          runtime_params,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=None,
          conductivity=mock.ANY,
      )

  def test_raises_error_if_conductivity_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource(
        model_func=ohmic_heat_source.ohmic_model_func
    )
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
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.',
    ):
      source.get_value(
          runtime_params,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=mock.ANY,
          conductivity=None,
      )

  @mock.patch.object(
      ohmic_heat_source.psi_calculations, 'calculate_psidot_from_psi_sources'
  )
  @mock.patch.object(ohmic_heat_source.psi_calculations, 'calc_j_total')
  def test_ohmic_power_with_prescribed_psidot(
      self, mock_calc_j_total, mock_calculate_psidot
  ):
    n_rho = 10
    j_total_val = jnp.ones(n_rho)
    mock_calc_j_total.return_value = (j_total_val, mock.Mock(), mock.Mock())

    prescribed_psidot = jnp.full((n_rho,), 2.0)

    runtime_params = mock.Mock()
    # Key conditions for using prescribed psidot
    runtime_params.numerics.evolve_current = False
    runtime_params.profile_conditions.psidot = prescribed_psidot

    # Mock geometry
    geo = mock.Mock()
    # Used in power calculation
    geo.R_major_profile = jnp.full((n_rho,), 5.0)

    # Mock other inputs
    core_profiles = mock.Mock()
    calculated_source_profiles = mock.Mock()
    calculated_source_profiles.total_psi_sources.return_value = jnp.zeros(n_rho)
    conductivity = mock.Mock()

    # Call function
    (pohm,) = ohmic_heat_source.ohmic_model_func(
        runtime_params=runtime_params,
        geo=geo,
        unused_source_name='ohmic',
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        conductivity=conductivity,
    )

    # Verify that the standard calculation was NOT called
    mock_calculate_psidot.assert_not_called()

    # Verify result matches calculation using prescribed psidot
    expected_pohm = jnp.abs(
        j_total_val * prescribed_psidot / (2 * jnp.pi * geo.R_major_profile)
    )
    np.testing.assert_allclose(pohm, expected_pohm)


if __name__ == '__main__':
  absltest.main()
