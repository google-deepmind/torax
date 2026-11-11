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
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.geometry import trapped_fraction


class CircularGeometryTest(parameterized.TestCase):

  @parameterized.parameters([
      trapped_fraction.TrappedFractionSource.FILE,
      trapped_fraction.TrappedFractionSource.EXACT,
  ])
  def test_trapped_fraction_source_not_supported(
      self, trapped_fraction_source: trapped_fraction.TrappedFractionSource
  ):
    """Tests that CircularConfig only supports SAUTER."""
    with self.assertRaisesRegex(ValueError, 'not supported for CircularConfig'):
      circular_geometry.CircularConfig(
          trapped_fraction_source=trapped_fraction_source
      )

  def test_build_time_dependent_geometry_from_circular(self):
    """Test that time-dependent circular geometry can be built."""
    cfg = circular_geometry.CircularConfig(
        n_rho=25,
        elongation_LCFS=1.72,
        R_major={0.0: 6.2, 10.0: 7.2},
        a_minor={0.0: 2.0, 10.0: 1.0},
        B_0=5.3,
        hires_factor=4,
    )
    geo = cfg.build_runtime_params(5.0).geometry
    np.testing.assert_allclose(geo.R_major, 6.7)
    np.testing.assert_allclose(geo.a_minor, 1.5)

  def test_circular_geometry_can_be_input_to_jitted_function(self):

    @jax.jit
    def foo(geo: geometry.Geometry):
      return geo.R_major

    geo = (
        circular_geometry.CircularConfig(
            n_rho=25,
            elongation_LCFS=1.72,
            R_major=6.2,
            a_minor=2.0,
            B_0=5.3,
            hires_factor=4,
        )
        .build_runtime_params(0.0)
        .geometry
    )
    # Make sure you can call the function with geo as an arg.
    foo(geo)

  def test_constant_circular_config_prescribed_model(self):
    cfg = circular_geometry.CircularConfig()
    params = cfg.build_runtime_params(0.0)
    geo = params.geometry
    np.testing.assert_allclose(geo.R_major, 6.2)

  def test_time_varying_circular_config_prescribed_model(self):
    cfg_with_phi_b = circular_geometry.CircularConfig(
        R_major={0.0: 6.2, 1.0: 7.0},
        a_minor={0.0: 2.0, 1.0: 2.0},
        calc_Phi_b_dot=True,
    )
    params = cfg_with_phi_b.build_runtime_params(0.5)
    geo = params.geometry
    np.testing.assert_allclose(geo.R_major, 6.6)
    self.assertTrue(params.calc_Phi_b_dot)

    @jax.jit
    def jitted_eval(cfg_t, t):
      return cfg_t.build_runtime_params(t)

    params_jit = jitted_eval(cfg_with_phi_b, 0.5)
    np.testing.assert_allclose(params_jit.geometry.R_major, 6.6)
    self.assertTrue(params_jit.calc_Phi_b_dot)

    cfg_without_phi_b = circular_geometry.CircularConfig(
        R_major={0.0: 6.2, 1.0: 7.0},
        a_minor={0.0: 2.0, 1.0: 2.0},
        calc_Phi_b_dot=False,
    )
    params_no_phi = cfg_without_phi_b.build_runtime_params(0.5)
    self.assertFalse(params_no_phi.calc_Phi_b_dot)
    params_no_phi_jit = jitted_eval(cfg_without_phi_b, 0.5)
    self.assertFalse(params_no_phi_jit.calc_Phi_b_dot)

  def test_circular_config_json_roundtrip(self):
    cfg = circular_geometry.CircularConfig(
        R_major={0.0: 6.2, 1.0: 7.0},
        a_minor=2.0,
    )
    json_str = cfg.model_dump_json()
    cfg_restored = circular_geometry.CircularConfig.model_validate_json(
        json_str
    )
    chex.assert_trees_all_equal(cfg.model_dump(), cfg_restored.model_dump())
    self.assertEqual(
        cfg_restored.build_runtime_params(0.0).geometry.geometry_type,
        geometry.GeometryType.CIRCULAR,
    )

  def test_circular_config_update_fields(self):
    cfg = circular_geometry.CircularConfig()
    cfg._update_fields({'R_major': 7.0})
    np.testing.assert_allclose(cfg.R_major.get_value(0.0), 7.0)
    np.testing.assert_allclose(
        cfg.build_runtime_params(0.0).geometry.R_major, 7.0
    )

  def test_circular_config_invalid_a_minor_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'a_minor must be less than or equal to R_major.'
    ):
      circular_geometry.CircularConfig(R_major=1.0, a_minor=2.0)

    with self.assertRaisesRegex(
        ValueError, 'a_minor must be less than or equal to R_major.'
    ):
      circular_geometry.CircularConfig(
          R_major={0.0: 2.0, 1.0: 1.0},
          a_minor={0.0: 1.0, 1.0: 1.5},
      )

  def test_build_circular_geometry_jit_and_grad(self):
    face_centers = np.linspace(0.0, 1.0, 25)

    def loss_fn(r_maj, a_min, b0, elong):
      geo = circular_geometry._build_circular_geometry(
          face_centers=face_centers,
          elongation_LCFS=elong,
          R_major=r_maj,
          a_minor=a_min,
          B_0=b0,
          hires_factor=4,
      )
      return jnp.sum(geo.vpr) + jnp.sum(geo.Phi) + jnp.sum(geo.volume)

    jax.test_util.check_grads(
        jax.jit(loss_fn),
        (6.2, 2.0, 5.3, 1.72),
        order=1,
        modes=['rev'],
    )


if __name__ == '__main__':
  absltest.main()
