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
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.neoclassical.conductivity import redl
from torax._src.neoclassical.conductivity import sauter
from torax._src.neoclassical.formulas import formulas


class RedlConductivityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.n_rho = 10
    self.geo = circular_geometry.CircularConfig(
        n_rho=self.n_rho
    ).build_geometry()
    self.core_profiles = mock.create_autospec(
        state.CoreProfiles,
        T_i=cell_variable.CellVariable(
            value=jnp.linspace(2.0, 0.5, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        T_e=cell_variable.CellVariable(
            value=jnp.linspace(2.0, 0.5, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        psi=cell_variable.CellVariable(
            value=jnp.linspace(9000, 4000, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        n_e=cell_variable.CellVariable(
            value=jnp.linspace(1e20, 0.5e20, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        n_i=cell_variable.CellVariable(
            value=jnp.linspace(0.8e20, 0.4e20, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        Z_i_face=jnp.ones(self.n_rho + 1),
        Z_eff_face=jnp.full(self.n_rho + 1, 2.0),
        q_face=jnp.linspace(1.0, 4.0, self.n_rho + 1),
    )

  def test_redl_conductivity_shape_and_positivity(self):
    model = redl.RedlModel()
    result = model.calculate_conductivity(self.geo, self.core_profiles)
    self.assertEqual(result.sigma.shape, (self.n_rho,))
    self.assertEqual(result.sigma_face.shape, (self.n_rho + 1,))
    self.assertTrue(np.all(result.sigma > 0.0))
    self.assertTrue(np.all(result.sigma_face > 0.0))

  def test_redl_conductivity_with_and_without_analytical_cache(self):
    model = redl.RedlModel()
    result_no_cache = model.calculate_conductivity(self.geo, self.core_profiles)
    cache = formulas.compute_analytical_cache(self.geo, self.core_profiles)
    result_with_cache = model.calculate_conductivity(
        self.geo, self.core_profiles, analytical_cache=cache
    )
    np.testing.assert_allclose(result_no_cache.sigma, result_with_cache.sigma)
    np.testing.assert_allclose(
        result_no_cache.sigma_face, result_with_cache.sigma_face
    )
    # Verify Redl conductivity differs from Sauter conductivity for Z_eff=2.0
    sauter_result = sauter.SauterModel().calculate_conductivity(
        self.geo, self.core_profiles, analytical_cache=cache
    )
    self.assertFalse(
        np.allclose(result_with_cache.sigma_face, sauter_result.sigma_face)
    )


if __name__ == '__main__':
  absltest.main()
