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
from torax._src.core_profiles import solver_profiles
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.neoclassical.conductivity import sauter
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas


class SauterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.n_rho = 10
    self.geo = circular_geometry.CircularConfig(
        n_rho=self.n_rho
    ).build_geometry()
    kinetic = mock.MagicMock(
        T_i=cell_variable.CellVariable(
            value=jnp.linspace(400, 700, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        T_e=cell_variable.CellVariable(
            value=jnp.linspace(4000, 7000, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        n_e=cell_variable.CellVariable(
            value=jnp.linspace(100, 200, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        n_i=cell_variable.CellVariable(
            value=jnp.linspace(100, 200, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        n_impurity=cell_variable.CellVariable(
            value=jnp.linspace(10, 20, self.n_rho),
            face_centers=self.geo.rho_face_norm,
            right_face_constraint=20.0,
            right_face_grad_constraint=None,
        ),
    )
    ions = mock.MagicMock(
        Z_i_face=jnp.linspace(1000, 2000, self.n_rho + 1),
        Z_eff_face=jnp.linspace(1.0, 1.0, self.n_rho + 1),
        impurity_fractions={},
    )
    flux = mock.MagicMock(
        psi=cell_variable.CellVariable(
            value=jnp.linspace(9000, 4000, self.n_rho),
            face_centers=self.geo.rho_face_norm,
        ),
        q_face=jnp.linspace(1, 5, self.n_rho + 1),
    )
    self.core_profiles = mock.create_autospec(
        solver_profiles.SolverProfiles,
        instance=True,
        kinetic=kinetic,
        ions=ions,
        flux=flux,
        impurity_density_scaling_face=jnp.ones(self.n_rho + 1),
    )
    self.neoclassical_intermediates = (
        formulas.compute_neoclassical_intermediates(
            self.geo, self.core_profiles
        )
    )

  def _calculate_conductivity(self, runtime_params: sauter.RuntimeParams):
    return sauter.SauterModel().calculate_conductivity(
        runtime_params,
        self.geo,
        self.core_profiles,
        self.neoclassical_intermediates,
    )

  def test_sauter_conductivity_current_is_correct_shape(self):
    result = self._calculate_conductivity(sauter.RuntimeParams())
    self.assertEqual(result.sigma.shape, (self.n_rho,))
    self.assertEqual(result.sigma_face.shape, (self.n_rho + 1,))

  @parameterized.parameters(0.5, 2.0)
  def test_conductivity_multiplier_scales_sigma(self, multiplier):
    reference = self._calculate_conductivity(sauter.RuntimeParams())
    result = self._calculate_conductivity(
        sauter.RuntimeParams(conductivity_multiplier=multiplier)
    )
    np.testing.assert_allclose(result.sigma, multiplier * reference.sigma)
    np.testing.assert_allclose(
        result.sigma_face, multiplier * reference.sigma_face
    )

  def test_neoclassical_correction_false_returns_spitzer_conductivity(self):
    neoclassical = self._calculate_conductivity(sauter.RuntimeParams())
    spitzer = self._calculate_conductivity(
        sauter.RuntimeParams(neoclassical_correction=False)
    )
    correction = sauter_formulas.calculate_L33(
        f_trap=self.neoclassical_intermediates.f_trap,
        nu_e_star=self.neoclassical_intermediates.nu_e_star,
        Z_eff=self.core_profiles.ions.Z_eff_face,
    )
    np.testing.assert_allclose(
        neoclassical.sigma_face, spitzer.sigma_face * correction
    )


if __name__ == "__main__":
  absltest.main()
