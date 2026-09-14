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

"""Unit tests for internal boundary conditions builder."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import builder
from torax._src.internal_boundary_conditions import internal_boundary_conditions as ibc_lib
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name


class BuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=20).build_geometry()
    self.core_profiles = mock.create_autospec(
        state.CoreProfiles, instance=True
    )
    self.inboard_mask = self.geo.rho_norm <= 0.3
    self.ped_cell = int(jnp.argmin(jnp.abs(self.geo.rho_norm - 0.9)))
    self.user_ibc = ibc_lib.InternalBoundaryConditions(
        T_i=jnp.zeros_like(self.geo.rho_norm),
        T_e=jnp.where(self.inboard_mask, 10.0, 0.0),
        n_e=jnp.zeros_like(self.geo.rho_norm),
    )

  def _make_runtime_params(
      self,
      t: float = 0.0,
      set_pedestal: bool = True,
      mode: pedestal_runtime_params_lib.Mode = (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
      ),
      use_formation_model: bool = True,
      transition_time_width: float = 1.0,
      user_ibc: ibc_lib.InternalBoundaryConditions | None = None,
  ) -> runtime_params_lib.RuntimeParams:
    pedestal = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams,
        instance=True,
        mode=mode,
        set_pedestal=set_pedestal,
        use_formation_model_with_internal_boundary_condition=use_formation_model,
        transition_time_width=transition_time_width,
        pedestal_profile_form=pedestal_runtime_params_lib.PedestalProfileForm.SET_AT_PED_TOP,
    )
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.ProfileConditions,
        instance=True,
        internal_boundary_conditions=user_ibc,
    )
    return mock.create_autospec(
        runtime_params_lib.RuntimeParams,
        instance=True,
        t=t,
        pedestal=pedestal,
        profile_conditions=profile_conditions,
    )

  def _make_transition_state(
      self,
      confinement_mode: pedestal_transition_state.ConfinementMode,
      T_i_ped: float = 2.0,
      T_e_ped: float = 2.0,
      n_e_ped: float = 1e19,
      rho_norm_ped_top: float = 0.9,
      transition_start_time: float = 0.0,
      T_e_ped_L_mode: float = 0.0,
  ) -> pedestal_transition_state.PedestalTransitionState:
    output = pedestal_model_output.PedestalModelOutput(
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
        rho_norm_ped_top=rho_norm_ped_top,
    )
    return pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(transition_start_time),
        T_i_ped_L_mode=jnp.array(0.0),
        T_e_ped_L_mode=jnp.array(T_e_ped_L_mode),
        n_e_ped_L_mode=jnp.array(0.0),
        confinement_mode=confinement_mode,
        pedestal_model_output=output,
        previous_pedestal_model_output=output,
    )

  def test_build_internal_boundary_conditions_mutual_exclusion(self):
    runtime_params = self._make_runtime_params(user_ibc=self.user_ibc)

    with self.subTest('l_mode_user_ibc_active_pedestal_disabled'):
      l_mode_state = self._make_transition_state(
          confinement_mode=pedestal_transition_state.ConfinementMode.L_MODE
      )
      built_ibc_l = builder.build_internal_boundary_conditions(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          pedestal_transition_state=l_mode_state,
      )
      # User IBC pins cell values in rho_norm <= 0.3 to 10.0 keV.
      self.assertTrue(jnp.any(built_ibc_l.T_e[self.inboard_mask] == 10.0))
      # Pedestal location (0.9) is unpinned (0.0) in L-mode.
      self.assertEqual(built_ibc_l.T_e[self.ped_cell], 0.0)

    with self.subTest('h_mode_pedestal_active_user_ibc_disabled'):
      h_mode_state = self._make_transition_state(
          confinement_mode=pedestal_transition_state.ConfinementMode.H_MODE,
          T_e_ped=2.0,
      )
      built_ibc_h = builder.build_internal_boundary_conditions(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          pedestal_transition_state=h_mode_state,
      )
      # User IBC region is completely turned off (0.0) in H-mode.
      self.assertTrue(jnp.all(built_ibc_h.T_e[self.inboard_mask] == 0.0))
      # Pedestal top cell is pinned to pedestal height (2.0).
      self.assertEqual(built_ibc_h.T_e[self.ped_cell], 2.0)

  def test_dynamic_transition_scales_pedestal_ibc(self):
    runtime_params = self._make_runtime_params(
        t=1.0,  # (1.0 - 0.0) / 2.0 = 0.5 ramp fraction
        transition_time_width=2.0,
        user_ibc=self.user_ibc,
    )
    transition_state = self._make_transition_state(
        confinement_mode=pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE,
        T_e_ped=4.0,
        T_e_ped_L_mode=0.0,
        transition_start_time=0.0,
    )
    built_ibc = builder.build_internal_boundary_conditions(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=transition_state,
    )
    # Scaled pedestal: 0.0 + 0.5 * (4.0 - 0.0) = 2.0 keV at ped_cell.
    self.assertAlmostEqual(float(built_ibc.T_e[self.ped_cell]), 2.0, places=5)
    # User IBC is disabled during transition.
    self.assertTrue(jnp.all(built_ibc.T_e[self.inboard_mask] == 0.0))

  def test_set_pedestal_false_preserves_user_ibc_in_h_mode(self):
    runtime_params = self._make_runtime_params(
        set_pedestal=False,
        user_ibc=self.user_ibc,
    )
    h_mode_state = self._make_transition_state(
        confinement_mode=pedestal_transition_state.ConfinementMode.H_MODE,
        T_e_ped=2.0,
    )
    built_ibc = builder.build_internal_boundary_conditions(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=h_mode_state,
    )
    # Pedestal is inactive, so user IBC is preserved.
    self.assertTrue(jnp.any(built_ibc.T_e[self.inboard_mask] == 10.0))
    self.assertEqual(built_ibc.T_e[self.ped_cell], 0.0)

  def test_none_user_ibc_handled_gracefully(self):
    runtime_params = self._make_runtime_params(user_ibc=None)
    l_mode_state = self._make_transition_state(
        confinement_mode=pedestal_transition_state.ConfinementMode.L_MODE
    )
    built_ibc = builder.build_internal_boundary_conditions(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_transition_state=l_mode_state,
    )
    self.assertTrue(jnp.all(built_ibc.T_e == 0.0))
    self.assertTrue(jnp.all(built_ibc.T_i == 0.0))
    self.assertTrue(jnp.all(built_ibc.n_e == 0.0))


if __name__ == '__main__':
  absltest.main()
