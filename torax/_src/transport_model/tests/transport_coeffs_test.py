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
from jax import numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.transport_model import transport_coeffs


class TransportCoeffsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    self.times = np.array([0.0])
    self.context = output_grid_context.OutputGridContext(
        times=self.times,
        rho_face_norm=self.geo.rho_face_norm,
        rho_cell_norm=self.geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], self.geo.rho_norm, [1.0]]
        ),
    )
    self.n_face = self.geo.rho_face_norm.size

  def test_zeros(self):
    coeffs = transport_coeffs.TransportCoeffs.zeros(self.geo)
    np.testing.assert_allclose(coeffs.chi_face_ion, np.zeros(self.n_face))
    np.testing.assert_allclose(coeffs.chi_face_el, np.zeros(self.n_face))
    np.testing.assert_allclose(coeffs.d_face_el, np.zeros(self.n_face))
    np.testing.assert_allclose(coeffs.v_face_el, np.zeros(self.n_face))

  def test_addition(self):
    c1 = transport_coeffs.TransportCoeffs(
        chi_face_ion=jnp.ones(self.n_face) * 1.0,
        chi_face_el=jnp.ones(self.n_face) * 2.0,
        d_face_el=jnp.ones(self.n_face) * 0.5,
        v_face_el=jnp.ones(self.n_face) * -0.1,
    )
    c2 = transport_coeffs.TransportCoeffs(
        chi_face_ion=jnp.ones(self.n_face) * 0.5,
        chi_face_el=jnp.ones(self.n_face) * 0.3,
        d_face_el=jnp.ones(self.n_face) * 0.1,
        v_face_el=jnp.ones(self.n_face) * -0.2,
    )
    c_sum = c1 + c2
    np.testing.assert_allclose(c_sum.chi_face_ion, np.ones(self.n_face) * 1.5)
    np.testing.assert_allclose(c_sum.chi_face_el, np.ones(self.n_face) * 2.3)
    np.testing.assert_allclose(c_sum.d_face_el, np.ones(self.n_face) * 0.6)
    np.testing.assert_allclose(c_sum.v_face_el, np.ones(self.n_face) * -0.3)

  def test_chi_max(self):
    coeffs = transport_coeffs.TransportCoeffs(
        chi_face_ion=jnp.ones(self.n_face) * 1.0,
        chi_face_el=jnp.ones(self.n_face) * 2.5,
        d_face_el=jnp.ones(self.n_face) * 0.5,
        v_face_el=jnp.ones(self.n_face) * -0.1,
    )
    chi_max = coeffs.chi_max(self.geo)
    expected = jnp.max(2.5 * self.geo.g1_over_vpr2_face)
    np.testing.assert_allclose(chi_max, expected)

  def test_to_output_dict_base(self):
    coeffs = transport_coeffs.TransportCoeffs(
        chi_face_ion=jnp.ones((1, self.n_face)) * 1.0,
        chi_face_el=jnp.ones((1, self.n_face)) * 2.0,
        d_face_el=jnp.ones((1, self.n_face)) * 0.5,
        v_face_el=jnp.ones((1, self.n_face)) * -0.1,
    )
    out = coeffs.to_output_dict(self.context)
    self.assertIn(output_keys.CHI_TURB_I, out)
    self.assertIn(output_keys.CHI_TURB_E, out)
    self.assertIn(output_keys.D_TURB_E, out)
    self.assertIn(output_keys.V_TURB_E, out)


if __name__ == '__main__':
  absltest.main()
