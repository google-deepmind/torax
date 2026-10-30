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

"""Unit tests for InternalBoundaryCondition and AdaptiveTransport models."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.pedestal_model.structure import pedestal_structure
from torax._src.physics import scaling_laws
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config as torax_pydantic_config

# pylint: disable=invalid-name,protected-access
ConfinementMode = pedestal_runtime_params_lib.ConfinementMode

_P_LH = 10.0
_HYSTERESIS = 0.8
_TRANSITION_WIDTH = 2.0


class InternalBoundaryConditionPedestalModelTest(parameterized.TestCase):
  """Unit tests for InternalBoundaryConditionPedestalModel."""

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'mode': 'internal_boundary_condition',
        'formation_model': {
            'model_name': 'martin_scaling',
            'P_LH_hysteresis_factor': _HYSTERESIS,
        },
        'transition_time_width': _TRANSITION_WIDTH,
        'structure_model': {
            'model_name': 'set_T_ped_n_ped',
            'T_i_ped': 4.5,
            'T_e_ped': 4.5,
            'n_e_ped': 0.62e20,
            'rho_norm_ped_top': 0.9,
        },
    }
    config['sources'] = {
        'generic_heat': {
            'gaussian_location': 0.15,
            'gaussian_width': 0.1,
            'P_total': 20.0e6,
            'electron_heat_fraction': 0.8,
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    self.pedestal_model = (
        pedestal_model.InternalBoundaryConditionPedestalModel(
            structure_model=(
                torax_config.pedestal.structure_model.build_structure_model()
            ),
            formation_model=(
                torax_config.pedestal.formation_model.build_formation_model()
            ),
        )
    )

    self.core_profiles, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_states = source_models.initial_state(self.runtime_params)
    self.source_profiles, _ = source_profile_builders.build_source_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    self.formation_state = (
        self.pedestal_model.formation_model.initial_state(
            self.runtime_params, self.geo, self.core_profiles
        )
    )
    self.structure_state = (
        self.pedestal_model.structure_model.initial_state(
            self.runtime_params, self.geo, self.core_profiles
        )
    )

  def _make_ibc_state(
      self,
      confinement_mode: ConfinementMode,
      transition_start_time: float = -jnp.inf,
      T_i_ped_L_mode: float = 0.5,
      T_e_ped_L_mode: float = 0.4,
      n_e_ped_L_mode: float = 0.5e19,
      rho_norm_ped_top: float = 0.9,
  ) -> pedestal_state_lib.IBCPedestalState:
    return pedestal_state_lib.IBCPedestalState(
        structure=self.structure_state,
        formation=self.formation_state,
        confinement_mode=jnp.array(confinement_mode),
        transition_start_time=jnp.array(transition_start_time),
        T_i_ped_L_mode=jnp.array(T_i_ped_L_mode),
        T_e_ped_L_mode=jnp.array(T_e_ped_L_mode),
        n_e_ped_L_mode=jnp.array(n_e_ped_L_mode),
        rho_norm_ped_top=jnp.array(rho_norm_ped_top),
    )

  def _call_pre_step(
      self,
      pedestal_state: pedestal_state_lib.IBCPedestalState,
      P_SOL: float,
      t: float = 5.0,
      explicit_pedestal: bool = False,
  ) -> tuple[
      pedestal_state_lib.IBCPedestalState,
      pedestal_structure.PedestalStructure | None,
  ]:
    pedestal_params = dataclasses.replace(
        self.runtime_params.pedestal,
        explicit_pedestal=explicit_pedestal,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, t=jnp.array(t), pedestal=pedestal_params
    )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      return self.pedestal_model.pre_step(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_state=pedestal_state,
      )

  def test_initialize_structure_state_threading(self):
    ped_top_idx = -3
    rho_norm_ped_top = self.geo.rho_norm[ped_top_idx]
    init_struct_state = pedestal_state_lib.StructureState()
    updated_struct_state = pedestal_state_lib.StructureState()
    structure_cls = type(self.pedestal_model.structure_model)
    with mock.patch.object(
        structure_cls,
        'initial_state',
        autospec=True,
        return_value=init_struct_state,
    ), mock.patch.object(
        structure_cls,
        '__call__',
        autospec=True,
        return_value=(
            pedestal_structure.PedestalStructure(
                rho_norm_ped_top=rho_norm_ped_top,
                T_i_ped=6.0,
                T_e_ped=6.0,
                n_e_ped=jnp.array(0.7e20),
            ),
            updated_struct_state,
        ),
    ) as mock_struct_call:
      init_output, init_state = self.pedestal_model.initialize(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
      )
    mock_struct_call.assert_called_once_with(
        self.pedestal_model.structure_model,
        self.runtime_params,
        self.geo,
        self.core_profiles,
        init_struct_state,
    )
    self.assertIs(init_state.structure, updated_struct_state)
    np.testing.assert_allclose(init_output.rho_norm_ped_top, rho_norm_ped_top)
    np.testing.assert_allclose(init_state.rho_norm_ped_top, rho_norm_ped_top)
    np.testing.assert_allclose(
        init_output.structure.T_i_ped,
        self.core_profiles.T_i.value[ped_top_idx],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='starting_in_L_mode',
          P_SOL=_P_LH * 0.5,
          custom_bc_multiplier=None,
          expected_confinement_mode=ConfinementMode.L_MODE,
          use_profile_values=True,
      ),
      dict(
          testcase_name='starting_in_H_mode_default_bc_multiplier',
          P_SOL=_P_LH * 1.5,
          custom_bc_multiplier=None,
          expected_confinement_mode=ConfinementMode.H_MODE,
          use_profile_values=False,
      ),
      dict(
          testcase_name='starting_in_H_mode_custom_bc_multiplier',
          P_SOL=_P_LH * 1.5,
          custom_bc_multiplier=3.5,
          expected_confinement_mode=ConfinementMode.H_MODE,
          use_profile_values=False,
      ),
  )
  def test_initialize_L_mode_baselines(
      self,
      P_SOL,
      custom_bc_multiplier,
      expected_confinement_mode,
      use_profile_values,
  ):
    """Verifies L-mode baselines when starting in L-mode vs H-mode."""
    runtime_params = self.runtime_params
    if custom_bc_multiplier is not None:
      custom_pedestal = dataclasses.replace(
          runtime_params.pedestal,
          initial_L_mode_bc_multiplier=custom_bc_multiplier,
      )
      runtime_params = dataclasses.replace(
          runtime_params, pedestal=custom_pedestal
      )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      init_output, init_state = self.pedestal_model.initialize(
          runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
      )
    self.assertEqual(init_output.confinement_mode, expected_confinement_mode)
    self.assertEqual(init_state.confinement_mode, expected_confinement_mode)
    if use_profile_values:
      ped_top_idx = jnp.argmin(
          jnp.abs(self.geo.rho_norm - init_output.rho_norm_ped_top)
      )
      expected_T_i = self.core_profiles.T_i.value[ped_top_idx]
      expected_T_e = self.core_profiles.T_e.value[ped_top_idx]
      expected_n_e = self.core_profiles.n_e.value[ped_top_idx]
    else:
      mult = 2.0 if custom_bc_multiplier is None else custom_bc_multiplier
      expected_T_i = mult * self.core_profiles.T_i.face_value()[-1]
      expected_T_e = mult * self.core_profiles.T_e.face_value()[-1]
      expected_n_e = mult * self.core_profiles.n_e.face_value()[-1]
    np.testing.assert_allclose(init_state.T_i_ped_L_mode, expected_T_i)
    np.testing.assert_allclose(init_state.T_e_ped_L_mode, expected_T_e)
    np.testing.assert_allclose(init_state.n_e_ped_L_mode, expected_n_e)

  @parameterized.named_parameters(
      dict(
          testcase_name='override_L_mode_above_P_LH_triggers_LH_ramp',
          initial_confinement_mode=ConfinementMode.L_MODE,
          P_SOL=_P_LH * 1.5,
          expected_init_mode=ConfinementMode.L_MODE,
          expected_update_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
      ),
      dict(
          testcase_name='override_H_mode_below_hysteresis_triggers_HL_ramp',
          initial_confinement_mode=ConfinementMode.H_MODE,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          expected_init_mode=ConfinementMode.H_MODE,
          expected_update_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
      ),
      dict(
          testcase_name='override_H_mode_in_hysteresis_band_stays_H_mode',
          initial_confinement_mode=ConfinementMode.H_MODE,
          P_SOL=_P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0),
          expected_init_mode=ConfinementMode.H_MODE,
          expected_update_mode=ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='inferred_in_hysteresis_band_starts_and_stays_L_mode',
          initial_confinement_mode=None,
          P_SOL=_P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0),
          expected_init_mode=ConfinementMode.L_MODE,
          expected_update_mode=ConfinementMode.L_MODE,
      ),
  )
  def test_initialize_initial_confinement_mode_override(
      self,
      initial_confinement_mode,
      P_SOL,
      expected_init_mode,
      expected_update_mode,
  ):
    pedestal_params = dataclasses.replace(
        self.runtime_params.pedestal,
        formation=dataclasses.replace(
            self.runtime_params.pedestal.formation,
            initial_confinement_mode=initial_confinement_mode,
        ),
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, pedestal=pedestal_params
    )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      init_output, init_state = self.pedestal_model.initialize(
          runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
      )
    self.assertEqual(init_output.confinement_mode, expected_init_mode)
    updated_state, _ = self._call_pre_step(init_state, P_SOL=P_SOL)
    self.assertEqual(updated_state.confinement_mode, expected_update_mode)

  @parameterized.named_parameters(
      dict(
          testcase_name='L_stays_L',
          init_mode=ConfinementMode.L_MODE,
          start_time=-np.inf,
          P_SOL=_P_LH * 0.5,
          t=5.0,
          expected_mode=ConfinementMode.L_MODE,
          expected_start_time=-np.inf,
      ),
      dict(
          testcase_name='L_to_transition_H',
          init_mode=ConfinementMode.L_MODE,
          start_time=-np.inf,
          P_SOL=_P_LH * 1.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          expected_start_time=5.0,
      ),
      dict(
          testcase_name='transition_H_incomplete',
          init_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          start_time=5.0 - _TRANSITION_WIDTH * 0.5,
          P_SOL=_P_LH * 1.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          expected_start_time=5.0 - _TRANSITION_WIDTH * 0.5,
      ),
      dict(
          testcase_name='transition_H_completes_to_H',
          init_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          start_time=5.0 - _TRANSITION_WIDTH - 0.1,
          P_SOL=_P_LH * 1.5,
          t=5.0,
          expected_mode=ConfinementMode.H_MODE,
          expected_start_time=5.0 - _TRANSITION_WIDTH - 0.1,
      ),
      dict(
          testcase_name='H_stays_H_in_hysteresis',
          init_mode=ConfinementMode.H_MODE,
          start_time=-np.inf,
          P_SOL=_P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0),
          t=5.0,
          expected_mode=ConfinementMode.H_MODE,
          expected_start_time=-np.inf,
      ),
      dict(
          testcase_name='H_stays_H_above_P_LH',
          init_mode=ConfinementMode.H_MODE,
          start_time=-np.inf,
          P_SOL=_P_LH * 1.5,
          t=5.0,
          expected_mode=ConfinementMode.H_MODE,
          expected_start_time=-np.inf,
      ),
      dict(
          testcase_name='H_to_transition_L',
          init_mode=ConfinementMode.H_MODE,
          start_time=-np.inf,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          expected_start_time=5.0,
      ),
      dict(
          testcase_name='transition_L_incomplete',
          init_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          start_time=5.0 - _TRANSITION_WIDTH * 0.5,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          expected_start_time=5.0 - _TRANSITION_WIDTH * 0.5,
      ),
      dict(
          testcase_name='transition_L_completes_to_L',
          init_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          start_time=5.0 - _TRANSITION_WIDTH - 0.1,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          t=5.0,
          expected_mode=ConfinementMode.L_MODE,
          expected_start_time=5.0 - _TRANSITION_WIDTH - 0.1,
      ),
      dict(
          testcase_name='dither_LH_to_HL',
          init_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          start_time=4.0,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          expected_start_time=2.0 * 5.0 - 4.0 - _TRANSITION_WIDTH,
      ),
      dict(
          testcase_name='dither_HL_to_LH',
          init_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          start_time=4.0,
          P_SOL=_P_LH * 1.5,
          t=5.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          expected_start_time=2.0 * 5.0 - 4.0 - _TRANSITION_WIDTH,
      ),
  )
  def test_pre_step_transitions_and_start_time(
      self,
      init_mode,
      start_time,
      P_SOL,
      t,
      expected_mode,
      expected_start_time,
  ):
    state = self._make_ibc_state(
        init_mode,
        transition_start_time=start_time,
    )
    new_state, explicit = self._call_pre_step(state, P_SOL=P_SOL, t=t)
    self.assertEqual(new_state.confinement_mode, expected_mode)
    self.assertIsNone(explicit)
    np.testing.assert_allclose(
        new_state.transition_start_time, expected_start_time
    )

  def test_L_to_H_captures_L_mode_values(self):
    ped_top_idx = -2
    rho_norm_ped_top = float(self.geo.rho_norm[ped_top_idx])
    state = self._make_ibc_state(
        ConfinementMode.L_MODE,
        T_i_ped_L_mode=0.0,
        T_e_ped_L_mode=0.0,
        n_e_ped_L_mode=0.0,
        rho_norm_ped_top=rho_norm_ped_top,
    )
    new_state, _ = self._call_pre_step(state, P_SOL=_P_LH * 1.5)
    np.testing.assert_allclose(
        new_state.T_i_ped_L_mode,
        self.core_profiles.T_i.value[ped_top_idx],
    )
    np.testing.assert_allclose(
        new_state.T_e_ped_L_mode,
        self.core_profiles.T_e.value[ped_top_idx],
    )
    np.testing.assert_allclose(
        new_state.n_e_ped_L_mode,
        self.core_profiles.n_e.value[ped_top_idx],
    )

  def test_non_L_to_H_preserves_L_mode_values(self):
    """Non LH transitions should keep existing L-mode values."""
    original_T_i = 0.5
    original_T_e = 0.4
    original_n_e = 0.5e19
    state = self._make_ibc_state(
        ConfinementMode.H_MODE,
        T_i_ped_L_mode=original_T_i,
        T_e_ped_L_mode=original_T_e,
        n_e_ped_L_mode=original_n_e,
    )
    new_state, _ = self._call_pre_step(
        state, P_SOL=_P_LH * _HYSTERESIS * 0.5
    )
    np.testing.assert_allclose(new_state.T_i_ped_L_mode, original_T_i)
    np.testing.assert_allclose(new_state.T_e_ped_L_mode, original_T_e)
    np.testing.assert_allclose(new_state.n_e_ped_L_mode, original_n_e)

  def test_zero_transition_time_width_steps_directly(self):
    ped_top_idx = -2
    rho_norm_ped_top = float(self.geo.rho_norm[ped_top_idx])
    zero_width_pedestal = dataclasses.replace(
        self.runtime_params.pedestal, transition_time_width=jnp.array(0.0)
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, t=jnp.array(5.0), pedestal=zero_width_pedestal
    )
    state = self._make_ibc_state(
        ConfinementMode.L_MODE,
        T_i_ped_L_mode=0.0,
        T_e_ped_L_mode=0.0,
        n_e_ped_L_mode=0.0,
        rho_norm_ped_top=rho_norm_ped_top,
    )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(_P_LH * 1.5),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      h_state, _ = self.pedestal_model.pre_step(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_state=state,
      )
    self.assertEqual(h_state.confinement_mode, ConfinementMode.H_MODE)
    np.testing.assert_allclose(
        h_state.T_i_ped_L_mode,
        self.core_profiles.T_i.value[ped_top_idx],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='lh_completion_yields_to_hl_exact_boundary',
          init_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          overshoot=0.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
      ),
      dict(
          testcase_name='lh_completion_yields_to_hl_overshoot_boundary',
          init_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          overshoot=0.3,
          expected_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
      ),
      dict(
          testcase_name='hl_completion_yields_to_lh_exact_boundary',
          init_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          P_SOL=_P_LH * 1.5,
          overshoot=0.0,
          expected_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
      ),
      dict(
          testcase_name='hl_completion_yields_to_lh_overshoot_boundary',
          init_mode=ConfinementMode.TRANSITIONING_TO_L_MODE,
          P_SOL=_P_LH * 1.5,
          overshoot=0.3,
          expected_mode=ConfinementMode.TRANSITIONING_TO_H_MODE,
      ),
  )
  def test_transition_completion_yields_to_reverse_trigger(
      self, init_mode, P_SOL, overshoot, expected_mode
  ):
    t = 5.0 + _TRANSITION_WIDTH + overshoot
    state = self._make_ibc_state(init_mode, transition_start_time=5.0)
    new_state, _ = self._call_pre_step(
        state,
        P_SOL=P_SOL,
        t=t,
    )
    self.assertEqual(new_state.confinement_mode, expected_mode)
    np.testing.assert_allclose(new_state.transition_start_time, t)

  def test_pre_step_implicit_pedestal_returns_none(self):
    state = self._make_ibc_state(ConfinementMode.L_MODE)
    _, explicit = self._call_pre_step(
        state, P_SOL=_P_LH * 0.5, explicit_pedestal=False
    )
    self.assertIsNone(explicit)

  def test_pre_step_explicit_pedestal_evaluates_structure(self):
    state = self._make_ibc_state(ConfinementMode.L_MODE)
    new_state, explicit = self._call_pre_step(
        state, P_SOL=_P_LH * 0.5, explicit_pedestal=True
    )
    self.assertIsNotNone(explicit)
    self.assertIsInstance(
        explicit, pedestal_structure.PedestalStructure
    )
    np.testing.assert_allclose(explicit.T_i_ped, 4.5)
    self.assertEqual(new_state.structure, state.structure)

  def test_call_skips_structure_model_when_explicit_provided(self):
    state = self._make_ibc_state(ConfinementMode.H_MODE)
    explicit_struct = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=jnp.array(0.88),
        T_i_ped=jnp.array(5.5),
        T_e_ped=jnp.array(5.5),
        n_e_ped=jnp.array(0.8e20),
    )
    structure_cls = type(self.pedestal_model.structure_model)
    with mock.patch.object(
        structure_cls, '__call__', autospec=True
    ) as mock_struct_call:
      out, updated_state = self.pedestal_model(
          runtime_params=self.runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_state=state,
          explicit_pedestal_structure=explicit_struct,
      )
      self.assertEqual(mock_struct_call.call_count, 0)
    np.testing.assert_allclose(out.structure.T_i_ped, 5.5)
    self.assertEqual(updated_state.structure, state.structure)
    np.testing.assert_allclose(updated_state.rho_norm_ped_top, 0.88)

  def test_call_without_explicit_pedestal(self):
    state = self._make_ibc_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE,
        transition_start_time=1.0,
        T_i_ped_L_mode=0.5,
        T_e_ped_L_mode=0.4,
        n_e_ped_L_mode=0.5e19,
    )
    t = 1.0 + 0.5 * _TRANSITION_WIDTH
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(t))
    formation_cls = type(self.pedestal_model.formation_model)
    expected_formation_state = (
        power_scaling_formation_model.PowerScalingFormationState(
            t=jnp.array(t)
        )
    )
    with mock.patch.object(
        formation_cls,
        'update_state',
        autospec=True,
        return_value=expected_formation_state,
    ) as mock_formation_call:
      out, updated_state = self.pedestal_model(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_state=state,
      )
    mock_formation_call.assert_called_once_with(
        self.pedestal_model.formation_model,
        runtime_params,
        self.geo,
        self.core_profiles,
        state.formation,
    )
    self.assertEqual(
        out.confinement_mode, ConfinementMode.TRANSITIONING_TO_H_MODE
    )
    np.testing.assert_allclose(out.rho_norm_ped_top, 0.9)
    np.testing.assert_allclose(updated_state.rho_norm_ped_top, 0.9)
    np.testing.assert_allclose(out.structure.T_i_ped, 0.5 + 0.5 * (4.5 - 0.5))
    np.testing.assert_allclose(out.structure.T_e_ped, 0.4 + 0.5 * (4.5 - 0.4))
    np.testing.assert_allclose(
        out.structure.n_e_ped, 0.5e19 + 0.5 * (0.62e20 - 0.5e19)
    )
    assert isinstance(
        updated_state.formation,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    assert updated_state.formation.t is not None
    np.testing.assert_allclose(updated_state.formation.t, t)


class AdaptiveTransportPedestalModelTest(parameterized.TestCase):
  """Unit tests for AdaptiveTransportPedestalModel."""

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'structure_model': {'model_name': 'set_T_ped_n_ped'},
        'mode': 'adaptive_transport',
        'formation_model': {
            'model_name': 'martin_scaling',
            'P_LH_hysteresis_factor': _HYSTERESIS,
        },
    }
    config['sources'] = {
        'generic_heat': {
            'gaussian_location': 0.15,
            'gaussian_width': 0.1,
            'P_total': 20.0e6,
            'electron_heat_fraction': 0.8,
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    self.pedestal_model = pedestal_model.AdaptiveTransportPedestalModel(
        structure_model=(
            torax_config.pedestal.structure_model.build_structure_model()
        ),
        formation_model=(
            torax_config.pedestal.formation_model.build_formation_model()
        ),
        saturation_model=(
            torax_config.pedestal.saturation_model.build_saturation_model()
        ),
    )

    self.core_profiles, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_states = source_models.initial_state(self.runtime_params)
    self.source_profiles, _ = source_profile_builders.build_source_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    self.formation_state = (
        self.pedestal_model.formation_model.initial_state(
            self.runtime_params, self.geo, self.core_profiles
        )
    )
    self.structure_state = (
        self.pedestal_model.structure_model.initial_state(
            self.runtime_params, self.geo, self.core_profiles
        )
    )

  def _make_adaptive_state(
      self,
      confinement_mode: ConfinementMode,
  ) -> pedestal_state_lib.AdaptiveTransportPedestalState:
    return pedestal_state_lib.AdaptiveTransportPedestalState(
        structure=self.structure_state,
        formation=self.formation_state,
        confinement_mode=jnp.array(confinement_mode),
    )

  def test_initialize(self):
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(_P_LH * 0.5),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      out, state = self.pedestal_model.initialize(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
      )
    self.assertIsInstance(
        out, pedestal_model_output.AdaptiveTransportPedestalModelOutput
    )
    self.assertIsInstance(
        state, pedestal_state_lib.AdaptiveTransportPedestalState
    )
    self.assertEqual(out.confinement_mode, ConfinementMode.L_MODE)
    self.assertEqual(state.confinement_mode, ConfinementMode.L_MODE)

  @parameterized.named_parameters(
      dict(
          testcase_name='L_stays_L',
          init_mode=ConfinementMode.L_MODE,
          P_SOL=_P_LH * 0.5,
          expected_mode=ConfinementMode.L_MODE,
      ),
      dict(
          testcase_name='L_to_H',
          init_mode=ConfinementMode.L_MODE,
          P_SOL=_P_LH * 1.5,
          expected_mode=ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='H_stays_H_above_threshold',
          init_mode=ConfinementMode.H_MODE,
          P_SOL=_P_LH * 1.5,
          expected_mode=ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='H_stays_H_in_hysteresis',
          init_mode=ConfinementMode.H_MODE,
          P_SOL=_P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0),
          expected_mode=ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='H_to_L',
          init_mode=ConfinementMode.H_MODE,
          P_SOL=_P_LH * _HYSTERESIS * 0.5,
          expected_mode=ConfinementMode.L_MODE,
      ),
  )
  def test_pre_step_confinement_mode_transitions(
      self, init_mode, P_SOL, expected_mode
  ):
    state = self._make_adaptive_state(init_mode)
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      new_state, _ = self.pedestal_model.pre_step(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
          state,
      )
    self.assertEqual(new_state.confinement_mode, expected_mode)

  def test_pre_step_implicit_pedestal_returns_none(self):
    state = self._make_adaptive_state(ConfinementMode.L_MODE)
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            explicit_pedestal=False,
        ),
    )
    _, explicit = self.pedestal_model.pre_step(
        runtime_params,
        self.geo,
        self.core_profiles,
        self.source_profiles,
        state,
    )
    self.assertIsNone(explicit)

  def test_pre_step_explicit_pedestal_evaluates_structure(self):
    state = self._make_adaptive_state(ConfinementMode.L_MODE)
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            explicit_pedestal=True,
        ),
    )
    new_state, explicit = self.pedestal_model.pre_step(
        runtime_params,
        self.geo,
        self.core_profiles,
        self.source_profiles,
        state,
    )
    self.assertIsNotNone(explicit)
    self.assertIsInstance(
        explicit, pedestal_structure.PedestalStructure
    )
    self.assertEqual(new_state.structure, state.structure)

  def test_call_with_explicit_pedestal(self):
    state = self._make_adaptive_state(ConfinementMode.H_MODE)
    explicit_struct = pedestal_structure.PedestalStructure(
        rho_norm_ped_top=jnp.array(0.9),
        T_i_ped=jnp.array(5.0),
        T_e_ped=jnp.array(5.0),
        n_e_ped=jnp.array(0.7e20),
    )
    structure_cls = type(self.pedestal_model.structure_model)
    with mock.patch.object(
        structure_cls, '__call__', autospec=True
    ) as mock_struct_call:
      call_out, call_state = self.pedestal_model(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
          state,
          explicit_pedestal_structure=explicit_struct,
      )
      self.assertEqual(mock_struct_call.call_count, 0)
    self.assertEqual(call_out.confinement_mode, ConfinementMode.H_MODE)
    np.testing.assert_allclose(call_out.structure.T_i_ped, 5.0)
    self.assertEqual(call_state.structure, state.structure)

  def test_call_without_explicit_pedestal(self):
    state = self._make_adaptive_state(ConfinementMode.H_MODE)
    formation_cls = type(self.pedestal_model.formation_model)
    expected_formation_state = (
        power_scaling_formation_model.PowerScalingFormationState(
            t=jnp.asarray(self.runtime_params.t)
        )
    )
    with mock.patch.object(
        formation_cls,
        'update_state',
        autospec=True,
        return_value=expected_formation_state,
    ) as mock_formation_call:
      out_implicit, state_implicit = self.pedestal_model(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
          state,
      )
    mock_formation_call.assert_called_once_with(
        self.pedestal_model.formation_model,
        self.runtime_params,
        self.geo,
        self.core_profiles,
        state.formation,
    )
    self.assertEqual(out_implicit.confinement_mode, ConfinementMode.H_MODE)
    self.assertIsInstance(
        out_implicit,
        pedestal_model_output.AdaptiveTransportPedestalModelOutput,
    )
    self.assertIsInstance(
        state_implicit,
        pedestal_state_lib.AdaptiveTransportPedestalState,
    )
    assert isinstance(
        state_implicit.formation,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    assert state_implicit.formation.t is not None
    np.testing.assert_allclose(
        state_implicit.formation.t, self.runtime_params.t
    )


if __name__ == '__main__':
  absltest.main()
