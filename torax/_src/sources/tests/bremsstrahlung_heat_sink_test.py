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
from typing import Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.sources import bremsstrahlung_heat_sink
from torax._src.sources.tests import test_lib
from torax._src.test_utils import torax_refs

# pylint: disable=invalid-name


class BremsstrahlungHeatSinkTest(test_lib.SingleProfileSourceTestCase):
  """Tests for BremsstrahlungHeatSink."""

  def setUp(self):
    super().setUp(
        source_config_class=bremsstrahlung_heat_sink.BremsstrahlungHeatSinkConfig,
        source_name=bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME,
    )

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_compare_against_known(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    references = references_getter()

    runtime_params, geo = references.get_runtime_params_and_geo()
    source_models = references.config.sources.build_models()
    neoclassical_models = references.config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    P_brem_total, P_brems_profile = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
        )
    )

    self.assertIsNotNone(P_brem_total)
    self.assertIsNotNone(P_brems_profile)

    P_brem_total_stott, P_brems_profile_stott = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
            use_relativistic_correction=True,
        )
    )

    self.assertIsNotNone(P_brem_total_stott)
    self.assertIsNotNone(P_brems_profile_stott)

    # Expect the relativistic correction to increase the total power.
    self.assertGreater(P_brem_total_stott, P_brem_total)

  def test_exclude_impurity_bremsstrahlung(self):
    """Tests that main-ion-only brems scales as Z_eff_main/Z_eff vs full."""
    n_rho = 25
    rho_face_norm = jnp.linspace(0, 1, n_rho + 1)

    # Set up synthetic profiles with Z_eff=2.0 and Z_i=1 (hydrogen).
    n_e_values = jnp.ones(n_rho) * 3e20
    T_e_values = jnp.linspace(10.0, 1.0, n_rho)  # keV, gradient
    Z_eff_face = jnp.ones(n_rho + 1) * 2.0
    Z_i_face = jnp.ones(n_rho + 1) * 1.0

    # For Z_eff=2 with Z_i=1, quasineutrality gives n_i/n_e = (Z_imp - Z_eff)
    # / (Z_imp - Z_i).
    # use Z_imp=10: n_i/n_e = (10-2)/(10-1) = 8/9.
    # Then Z_eff_main = n_i * Z_i^2 / n_e = 8/9.
    n_i_over_n_e = 8.0 / 9.0
    n_i_values = n_e_values * n_i_over_n_e

    n_e = cell_variable.CellVariable(
        value=n_e_values,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(3e20),
        face_centers=rho_face_norm,
    )
    T_e = cell_variable.CellVariable(
        value=T_e_values,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(1.0),
        face_centers=rho_face_norm,
    )
    n_i = cell_variable.CellVariable(
        value=n_i_values,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(3e20 * n_i_over_n_e),
        face_centers=rho_face_norm,
    )

    core_profiles = mock.MagicMock()
    core_profiles.n_e = n_e
    core_profiles.T_e = T_e
    core_profiles.n_i = n_i
    core_profiles.Z_i_face = Z_i_face
    core_profiles.Z_eff_face = Z_eff_face

    # Mock geometry — needs rho_norm, vpr, drho_norm for volume_integration.
    rho_norm = geometry_lib.face_to_cell(rho_face_norm)
    geo = mock.MagicMock()
    geo.rho_face_norm = rho_face_norm
    geo.rho_norm = rho_norm
    geo.vpr = jnp.ones(n_rho)
    geo.drho_norm = rho_face_norm[1] - rho_face_norm[0]

    # Full bremsstrahlung (using Z_eff).
    _, P_profile_full = bremsstrahlung_heat_sink.calc_bremsstrahlung(
        core_profiles, geo,
    )
    # Main-ion-only bremsstrahlung (using Z_eff_main = n_i * Z_i^2 / n_e).
    _, P_profile_main = bremsstrahlung_heat_sink.calc_bremsstrahlung(
        core_profiles, geo, exclude_impurity_bremsstrahlung=True,
    )

    expected_ratio = n_i_over_n_e / 2.0
    np.testing.assert_allclose(
        np.asarray(P_profile_main) / np.asarray(P_profile_full),
        expected_ratio,
        rtol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
