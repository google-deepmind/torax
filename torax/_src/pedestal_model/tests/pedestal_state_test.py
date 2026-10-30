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

"""Unit tests for pedestal_state."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.pedestal_model import pedestal_state
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from torax._src.pedestal_model.structure import pedestal_structure

# pylint: disable=invalid-name
ConfinementMode = runtime_params_lib.ConfinementMode


class PedestalStateTest(parameterized.TestCase):
  """Unit tests for PedestalState."""

  @parameterized.parameters(
      (ConfinementMode.L_MODE, False),
      (ConfinementMode.H_MODE, True),
      (ConfinementMode.TRANSITIONING_TO_H_MODE, True),
      (ConfinementMode.TRANSITIONING_TO_L_MODE, True),
  )
  def test_is_active(self, confinement_mode, expected_active):
    state = pedestal_state.PedestalState(
        confinement_mode=jnp.asarray(
            confinement_mode, dtype=jax_utils.get_int_dtype()
        )
    )
    self.assertEqual(bool(state.is_active), expected_active)


class IBCPedestalStateTest(parameterized.TestCase):
  """Unit tests for IBCPedestalState ramp calculation and structure scaling."""

  @parameterized.named_parameters(
      dict(testcase_name="before_start", t=0.5, width=1.0, expected=0.0),
      dict(testcase_name="at_start", t=1.0, width=1.0, expected=0.0),
      dict(testcase_name="midpoint", t=1.5, width=1.0, expected=0.5),
      dict(testcase_name="at_end", t=2.0, width=1.0, expected=1.0),
      dict(testcase_name="after_end", t=2.5, width=1.0, expected=1.0),
      dict(
          testcase_name="zero_width_at_start", t=1.0, width=0.0, expected=1.0
      ),
      dict(
          testcase_name="zero_width_before_start",
          t=0.5,
          width=0.0,
          expected=1.0,
      ),
  )
  def test_compute_ramp_fraction(self, t, width, expected):
    state = pedestal_state.IBCPedestalState(
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype())
    )
    self.assertEqual(
        state.compute_ramp_fraction(t=t, transition_time_width=width),
        expected,
    )

  def test_compute_ramp_fraction_zero_width_grad_is_finite(self):
    state = pedestal_state.IBCPedestalState(
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype())
    )
    grad_fn = jax.grad(
        lambda w: state.compute_ramp_fraction(
            t=jnp.asarray(1.5, dtype=jax_utils.get_dtype()),
            transition_time_width=w,
        )
    )
    grad_val = grad_fn(jnp.asarray(0.0, dtype=jax_utils.get_dtype()))
    self.assertTrue(bool(jnp.isfinite(grad_val)))

  def test_scale_structure_l_mode(self):
    state = pedestal_state.IBCPedestalState(
        confinement_mode=jnp.asarray(
            ConfinementMode.L_MODE, dtype=jax_utils.get_int_dtype()
        ),
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.asarray(3.0, dtype=jax_utils.get_dtype()),
    )
    structure = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=0.9,
        T_i_ped=5.0,
        T_e_ped=6.0,
        n_e_ped=7.0,
    )
    scaled = state.scale_structure(
        structure=structure,
        t=jnp.asarray(1.5, dtype=jax_utils.get_dtype()),
        transition_time_width=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
    )
    np.testing.assert_allclose(scaled.T_i_ped, 1.0)
    np.testing.assert_allclose(scaled.T_e_ped, 2.0)
    np.testing.assert_allclose(scaled.n_e_ped, 3.0)
    np.testing.assert_allclose(scaled.rho_norm_ped_top, 0.9)

  def test_scale_structure_h_mode(self):
    state = pedestal_state.IBCPedestalState(
        confinement_mode=jnp.asarray(
            ConfinementMode.H_MODE, dtype=jax_utils.get_int_dtype()
        ),
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.asarray(3.0, dtype=jax_utils.get_dtype()),
    )
    structure = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=0.9,
        T_i_ped=5.0,
        T_e_ped=6.0,
        n_e_ped=7.0,
    )
    scaled = state.scale_structure(
        structure=structure,
        t=jnp.asarray(1.5, dtype=jax_utils.get_dtype()),
        transition_time_width=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
    )
    np.testing.assert_allclose(scaled.T_i_ped, 5.0)
    np.testing.assert_allclose(scaled.T_e_ped, 6.0)
    np.testing.assert_allclose(scaled.n_e_ped, 7.0)
    np.testing.assert_allclose(scaled.rho_norm_ped_top, 0.9)

  def test_scale_structure_transitioning_to_h_mode(self):
    state = pedestal_state.IBCPedestalState(
        confinement_mode=jnp.asarray(
            ConfinementMode.TRANSITIONING_TO_H_MODE,
            dtype=jax_utils.get_int_dtype(),
        ),
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.asarray(3.0, dtype=jax_utils.get_dtype()),
    )
    structure = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=0.85,
        T_i_ped=5.0,
        T_e_ped=6.0,
        n_e_ped=7.0,
    )
    # Elapsed is 0.5 s with width 2.0 s -> ramp fraction is 0.25.
    # Scaled values: l + 0.25 * (h - l).
    scaled = state.scale_structure(
        structure=structure,
        t=jnp.asarray(1.5, dtype=jax_utils.get_dtype()),
        transition_time_width=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
    )
    np.testing.assert_allclose(scaled.T_i_ped, 2.0)
    np.testing.assert_allclose(scaled.T_e_ped, 3.0)
    np.testing.assert_allclose(scaled.n_e_ped, 4.0)
    np.testing.assert_allclose(scaled.rho_norm_ped_top, 0.85)

  def test_scale_structure_transitioning_to_l_mode(self):
    state = pedestal_state.IBCPedestalState(
        confinement_mode=jnp.asarray(
            ConfinementMode.TRANSITIONING_TO_L_MODE,
            dtype=jax_utils.get_int_dtype(),
        ),
        transition_start_time=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.asarray(1.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.asarray(3.0, dtype=jax_utils.get_dtype()),
    )
    structure = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=0.85,
        T_i_ped=5.0,
        T_e_ped=6.0,
        n_e_ped=7.0,
    )
    # Elapsed is 0.5 s with width 2.0 s -> ramp fraction is 0.25.
    # Scaled values: h + 0.25 * (l - h).
    scaled = state.scale_structure(
        structure=structure,
        t=jnp.asarray(1.5, dtype=jax_utils.get_dtype()),
        transition_time_width=jnp.asarray(2.0, dtype=jax_utils.get_dtype()),
    )
    np.testing.assert_allclose(scaled.T_i_ped, 4.0)
    np.testing.assert_allclose(scaled.T_e_ped, 5.0)
    np.testing.assert_allclose(scaled.n_e_ped, 6.0)
    np.testing.assert_allclose(scaled.rho_norm_ped_top, 0.85)


if __name__ == "__main__":
  absltest.main()
