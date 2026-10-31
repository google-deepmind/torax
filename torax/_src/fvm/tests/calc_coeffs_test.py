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

import copy
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import profiles
from torax._src.core_profiles import solver_profiles
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.fvm import calc_coeffs
from torax._src.fvm import discrete_system
from torax._src.geometry import geometry
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import config


@dataclasses.dataclass(frozen=True)
class _CalcCoeffsInputs:
  core_profiles: profiles.CoreProfiles
  runtime_params: runtime_params_lib.RuntimeParams
  geo: geometry.Geometry
  models: models_lib.Models
  explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs
  model_state: model_state_lib.ModelState


def _make_calc_coeffs_inputs(
    torax_config: config.ToraxConfig,
) -> _CalcCoeffsInputs:
  models = torax_config.build_models()
  core_profiles, runtime_params, geo = (
      core_profile_helpers.make_initial_inputs(torax_config)
  )
  source_states = models.source_models.initial_state(runtime_params)
  explicit_source_outputs, _ = source_profile_builders.build_basic_sources(
      source_models=models.source_models,
      source_states=source_states,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=True,
  )
  _, pedestal_state = models.pedestal_model.initialize(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_profiles=source_profiles.SourceProfiles.zeros(geo),
  )
  model_state = model_state_lib.ModelState(
      pedestal=pedestal_state,
      transport=models.transport_model.initial_state(runtime_params),
      sources=source_states,
  )
  explicit_model_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
      sources=explicit_source_outputs,
      pedestal=None,
  )
  return _CalcCoeffsInputs(
      core_profiles=core_profiles,
      runtime_params=runtime_params,
      geo=geo,
      models=models,
      explicit_model_outputs=explicit_model_outputs,
      model_state=model_state,
  )


class CalcCoeffsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=4, theta_implicit=0, pedestal_active=False),
      dict(num_cells=4, theta_implicit=0, pedestal_active=True),
      dict(num_cells=4, theta_implicit=0.5, pedestal_active=False),
      dict(num_cells=4, theta_implicit=0.5, pedestal_active=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_implicit, pedestal_active
  ):
    sources_config = default_sources.get_default_source_config()
    sources_config['ei_exchange']['Qei_multiplier'] = 0.0
    sources_config['generic_heat']['P_total'] = 0.0
    sources_config['fusion']['mode'] = 'zero'
    sources_config['ohmic']['mode'] = 'zero'
    torax_config = config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(
                I_p=15e6,
                I_p_source='profile_conditions',
                initial_psi_source='j',
            ),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(
                structure_model=dict(model_name='set_T_ped_n_ped'),
                formation_model=dict(
                    model_name='prescribed',
                    pedestal_active=pedestal_active,
                ),
            ),
            sources=sources_config,
            solver=dict(use_fixed_point=False, theta_implicit=theta_implicit),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(model_name='prescribed', chi_i=1)
                ),
                chi_min=0,
            ),
            time_step_calculator=dict(),
        )
    )
    inputs = _make_calc_coeffs_inputs(torax_config)
    calc_coeffs.calc_coeffs(
        runtime_params=inputs.runtime_params,
        geo=inputs.geo,
        solver_state=solver_state_lib.SolverState.from_core_profiles(
            inputs.core_profiles
        ),
        models=inputs.models,
        explicit_model_outputs=inputs.explicit_model_outputs,
        evolving_names=('T_i',),
        use_pereverzev=False,
        model_state=inputs.model_state,
    )

  def test_calc_coeffs_hash(self):

    def create_coeffs_callback(
        torax_config: config.ToraxConfig,
    ) -> calc_coeffs.CoeffsCallback:
      models = torax_config.build_models()
      evolving_names = tuple(['T_i'])
      return calc_coeffs.CoeffsCallback(
          models=models,
          evolving_names=evolving_names,
      )

    config_dict = {
        'pedestal': {},
        'transport': {},
        'solver': {},
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
        },
        'numerics': {},
        'sources': {},
        'geometry': {'geometry_type': 'circular'},
        'plasma_composition': {},
    }
    config_dict_with_pedestal = copy.deepcopy(config_dict)
    config_dict_with_pedestal['pedestal'] = {
        'structure_model': {'model_name': 'set_T_ped_n_ped'}
    }

    with self.subTest('same_coeffs_callback_hash_equal'):
      self.assertEqual(
          hash(
              create_coeffs_callback(config.ToraxConfig.from_dict(config_dict))
          ),
          hash(
              create_coeffs_callback(config.ToraxConfig.from_dict(config_dict))
          ),
      )
    with self.subTest('same_coeffs_callback_equal'):
      self.assertEqual(
          create_coeffs_callback(config.ToraxConfig.from_dict(config_dict)),
          create_coeffs_callback(config.ToraxConfig.from_dict(config_dict)),
      )
    with self.subTest('different_coeffs_callback_hash_not_equal'):
      self.assertNotEqual(
          hash(
              create_coeffs_callback(
                  config.ToraxConfig.from_dict(config_dict_with_pedestal)
              )
          ),
          hash(
              create_coeffs_callback(config.ToraxConfig.from_dict(config_dict))
          ),
      )
    with self.subTest('different_coeffs_callback_not_equal'):
      self.assertNotEqual(
          create_coeffs_callback(config.ToraxConfig.from_dict(config_dict)),
          create_coeffs_callback(
              config.ToraxConfig.from_dict(config_dict_with_pedestal)
          ),
      )

  @parameterized.parameters([
      dict(mesh_power=1.0, use_neumann_right_bc=False),
      dict(mesh_power=3.0, use_neumann_right_bc=False),
      dict(mesh_power=3.0, use_neumann_right_bc=True),
  ])
  def test_pereverzev_stationary_cancellation(
      self, mesh_power: float, use_neumann_right_bc: bool
  ):
    s = np.linspace(0.0, 1.0, 11)
    face_centers = s**mesh_power
    sources_config = default_sources.get_default_source_config()
    base_cfg = dict(
        numerics=dict(),
        plasma_composition=dict(),
        profile_conditions=dict(
            I_p=15e6,
            I_p_source='profile_conditions',
            initial_psi_source='j',
            T_i={0.0: {0.0: 20.0, 0.7: 8.0, 0.9: 5.0, 1.0: 0.2}},
            T_e={0.0: {0.0: 18.0, 0.7: 7.0, 0.9: 4.5, 1.0: 0.2}},
            n_e={0.0: {0.0: 1.5e20, 0.85: 1.1e20, 1.0: 0.4e20}},
        ),
        geometry=dict(
            geometry_type='circular',
            face_centers=face_centers,
        ),
        pedestal=dict(),
        sources=sources_config,
        solver=dict(
            solver_type='linear',
            use_pereverzev=False,
            chi_pereverzev=30.0,
            D_pereverzev=15.0,
        ),
        transport=dict(
            core_transport_models=dict(
                prescribed=dict(
                    model_name='prescribed',
                    chi_i=1.0,
                    chi_e=1.2,
                    D_e=0.5,
                    V_e=-0.8,
                )
            ),
        ),
        time_step_calculator=dict(),
    )
    per_cfg = copy.deepcopy(base_cfg)
    per_cfg['solver'] = dict(
        solver_type='linear',
        use_pereverzev=True,
        chi_pereverzev=30.0,
        D_pereverzev=15.0,
    )

    def _eval_rhs(config_dict):
      torax_config = config.ToraxConfig.from_dict(config_dict)
      inputs = _make_calc_coeffs_inputs(torax_config)
      core_profiles = inputs.core_profiles
      if use_neumann_right_bc:
        core_profiles = dataclasses.replace(
            core_profiles,
            kinetic=dataclasses.replace(
                core_profiles.kinetic,
                T_i=dataclasses.replace(
                    core_profiles.kinetic.T_i,
                    right_face_constraint=None,
                    right_face_grad_constraint=jnp.array(-5.0),
                ),
                T_e=dataclasses.replace(
                    core_profiles.kinetic.T_e,
                    right_face_constraint=None,
                    right_face_grad_constraint=jnp.array(-4.5),
                ),
            ),
        )
      evolving = ('T_i', 'T_e', 'n_e')
      coeffs = calc_coeffs.calc_coeffs(
          runtime_params=inputs.runtime_params,
          geo=inputs.geo,
          solver_state=solver_state_lib.SolverState.from_core_profiles(
              core_profiles
          ),
          explicit_model_outputs=inputs.explicit_model_outputs,
          evolving_names=evolving,
          use_pereverzev=config_dict['solver']['use_pereverzev'],
          models=inputs.models,
          model_state=inputs.model_state,
      )
      x_vars = (
          core_profiles.kinetic.T_i,
          core_profiles.kinetic.T_e,
          core_profiles.kinetic.n_e,
      )
      c_mat, c_vec = discrete_system.calc_c(x_vars, coeffs)
      x_vec = jnp.stack([v.value for v in x_vars], axis=-1)
      return np.asarray(c_mat.matvec(x_vec) + c_vec)

    rhs_base = _eval_rhs(base_cfg)
    rhs_per = _eval_rhs(per_cfg)
    for ch_idx in range(3):
      scale = np.max(np.abs(rhs_base[:, ch_idx]))
      np.testing.assert_allclose(
          (rhs_per[:, ch_idx] - rhs_base[:, ch_idx]) / scale,
          0.0,
          atol=1e-12,
      )

  def test_phi_b_dot_terms_match_analytical_rhs(self):
    torax_config = config.ToraxConfig.from_dict(
        dict(
            numerics=dict(),
            plasma_composition=dict(),
            profile_conditions=dict(
                I_p=15e6,
                I_p_source='profile_conditions',
                initial_psi_source='j',
                T_i={0.0: {0.0: 5.0, 1.0: 5.0}},
                T_e={0.0: {0.0: 5.0, 1.0: 5.0}},
                n_e={0.0: {0.0: 1.0e20, 1.0: 1.0e20}},
                normalize_n_e_to_nbar=False,
            ),
            geometry=dict(geometry_type='circular', elongation_LCFS=1.0),
            pedestal=dict(),
            sources=dict(ei_exchange=dict(Qei_multiplier=0.0)),
            solver=dict(),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(model_name='prescribed', V_e=0.0)
                ),
            ),
            time_step_calculator=dict(),
        )
    )
    inputs = _make_calc_coeffs_inputs(torax_config)
    geo = dataclasses.replace(inputs.geo, Phi_b_dot=2.5)
    evolving_names = ('T_i', 'T_e', 'n_e')
    coeffs = calc_coeffs.calc_coeffs(
        runtime_params=inputs.runtime_params,
        geo=geo,
        solver_state=solver_state_lib.SolverState.from_core_profiles(
            inputs.core_profiles
        ),
        models=inputs.models,
        explicit_model_outputs=inputs.explicit_model_outputs,
        evolving_names=evolving_names,
        use_pereverzev=False,
        model_state=inputs.model_state,
    )
    x = convertors.solver_state_to_solver_x_tuple(
        solver_state_lib.SolverState.from_core_profiles(inputs.core_profiles),
        evolving_names,
    )
    c_mat, c_vec = discrete_system.calc_c(x, coeffs)
    rhs = c_mat.matvec(jnp.stack([v.value for v in x], axis=-1)) + c_vec

    # With uniform profiles, diffusion vanishes and only the Phi_b_dot terms
    # remain. For circular geometry (vpr ~ rho_norm), the heat RHS
    # 0.75 * (Phi_b_dot / Phi_b) * vpr**(-2/3)
    #   * d/drhon(rhon * vpr**(5/3) * n * T)
    # reduces to 2 * (Phi_b_dot / Phi_b) * vpr * n * T and the particle RHS
    # 0.5 * (Phi_b_dot / Phi_b) * d/drhon(rhon * vpr * n_e) to
    # (Phi_b_dot / Phi_b) * vpr * n_e.
    phib_dot_vpr = geo.Phi_b_dot / geo.Phi_b * geo.vpr
    heat_prefactor = 2.0 * constants.CONSTANTS.keV_to_J
    n_i, n_e = (
        inputs.core_profiles.kinetic.n_i.value,
        inputs.core_profiles.kinetic.n_e.value,
    )
    expected = {
        'T_i': (
            heat_prefactor
            * phib_dot_vpr
            * n_i
            * inputs.core_profiles.kinetic.T_i.value
        ),
        'T_e': (
            heat_prefactor
            * phib_dot_vpr
            * n_e
            * inputs.core_profiles.kinetic.T_e.value
        ),
        'n_e': phib_dot_vpr * n_e,
    }
    for i, name in enumerate(evolving_names):
      with self.subTest(name):
        np.testing.assert_allclose(
            rhs[:, i] * convertors.STATE_SCALING_FACTORS[name],
            expected[name],
            rtol=1e-10,
        )

  def test_ware_pinch_responds_to_current_iterate_psidot(self):
    sources_config = default_sources.get_default_source_config()
    sources_config['ei_exchange']['Qei_multiplier'] = 0.0
    torax_config = config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_current=True),
            plasma_composition=dict(),
            profile_conditions=dict(I_p=15e6, initial_psi_source='j'),
            geometry=dict(geometry_type='circular'),
            pedestal=dict(),
            neoclassical=dict(
                model_name='analytical',
                transport=dict(model_name='angioni_sauter'),
            ),
            sources=sources_config,
            solver=dict(),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(
                        model_name='prescribed',
                        chi_i=1.0,
                        chi_e=1.0,
                        D_e=1.0,
                        V_e=0.0,
                    )
                ),
            ),
            time_step_calculator=dict(),
        )
    )
    inputs = _make_calc_coeffs_inputs(torax_config)

    core_profiles_zero_psidot = dataclasses.replace(
        inputs.core_profiles,
        psidot=dataclasses.replace(
            inputs.core_profiles.psidot,
            value=jnp.zeros_like(inputs.core_profiles.psidot.value),
            right_face_constraint=jnp.array(0.0),
            right_face_grad_constraint=None,
        ),
    )
    core_profiles_dummy_psidot = dataclasses.replace(
        inputs.core_profiles,
        psidot=dataclasses.replace(
            inputs.core_profiles.psidot,
            value=jnp.ones_like(inputs.core_profiles.psidot.value) * 100.0,
            right_face_constraint=jnp.array(100.0),
            right_face_grad_constraint=None,
        ),
    )
    coeffs_zero = calc_coeffs.calc_coeffs(
        runtime_params=inputs.runtime_params,
        geo=inputs.geo,
        solver_state=solver_state_lib.SolverState.from_core_profiles(
            core_profiles_zero_psidot
        ),
        models=inputs.models,
        explicit_model_outputs=inputs.explicit_model_outputs,
        evolving_names=('n_e',),
        use_pereverzev=False,
        model_state=inputs.model_state,
    )
    coeffs_dummy = calc_coeffs.calc_coeffs(
        runtime_params=inputs.runtime_params,
        geo=inputs.geo,
        solver_state=solver_state_lib.SolverState.from_core_profiles(
            core_profiles_dummy_psidot
        ),
        models=inputs.models,
        explicit_model_outputs=inputs.explicit_model_outputs,
        evolving_names=('n_e',),
        use_pereverzev=False,
        model_state=inputs.model_state,
    )

    with self.subTest('independent_of_incoming_psidot'):
      assert coeffs_zero.v_face is not None
      assert coeffs_dummy.v_face is not None
      # calc_coeffs recalculates psidot for the current iterate, so the
      # resulting particle convection does not depend on incoming
      # core_profiles.psidot.
      np.testing.assert_allclose(
          np.asarray(coeffs_zero.v_face[0]),
          np.asarray(coeffs_dummy.v_face[0]),
          rtol=1e-12,
      )

    # When psi is scaled, current-iterate psidot changes and particle convection
    # (Ware pinch) responds accordingly, even when incoming psidot was zero.
    core_profiles = inputs.core_profiles
    right_face_constraint = (
        core_profiles.flux.psi.right_face_constraint * 2.0
        if core_profiles.flux.psi.right_face_constraint is not None
        else None
    )
    right_face_grad_constraint = (
        core_profiles.flux.psi.right_face_grad_constraint * 2.0
        if core_profiles.flux.psi.right_face_grad_constraint is not None
        else None
    )
    core_profiles_scaled_psi = dataclasses.replace(
        core_profiles_zero_psidot,
        flux=dataclasses.replace(
            core_profiles_zero_psidot.flux,
            psi=dataclasses.replace(
                core_profiles.flux.psi,
                value=core_profiles.flux.psi.value * 2.0,
                right_face_constraint=right_face_constraint,
                right_face_grad_constraint=right_face_grad_constraint,
            ),
        ),
    )
    coeffs_scaled = calc_coeffs.calc_coeffs(
        runtime_params=inputs.runtime_params,
        geo=inputs.geo,
        solver_state=solver_state_lib.SolverState.from_core_profiles(
            core_profiles_scaled_psi
        ),
        models=inputs.models,
        explicit_model_outputs=inputs.explicit_model_outputs,
        evolving_names=('n_e',),
        use_pereverzev=False,
        model_state=inputs.model_state,
    )
    with self.subTest('responds_to_scaled_psi'):
      assert coeffs_scaled.v_face is not None
      self.assertFalse(
          np.allclose(
              np.asarray(coeffs_scaled.v_face[0]),
              np.asarray(coeffs_zero.v_face[0]),
          )
      )

  def test_calc_coeffs_updates_fast_ions_before_neoclassical(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 4}
    config_dict['numerics'] = {
        'evolve_current': True,
        'enable_fast_ions': True,
    }
    config_dict['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'He3': 0.03},
        },
    }
    config_dict['sources'] = {
        'icrh': {
            'model_name': 'scaled_profile',
            'minority_species': 'He3',
            'is_explicit': False,
        },
    }
    config_dict['profile_conditions']['fast_ions'] = [
        {
            'source': 'icrh',
            'species': 'He3',
            'n': {0.0: {0.0: 1e19, 1.0: 1e19}},
            'T': {0.0: {0.0: 100.0, 1.0: 100.0}},
            'n_right_bc': 0.0,
            'T_right_bc': 0.0,
        }
    ]
    torax_config = config.ToraxConfig.from_dict(config_dict)
    inputs = _make_calc_coeffs_inputs(torax_config)
    # Stale core_profiles with empty fast ions.
    stale_core_profiles = dataclasses.replace(inputs.core_profiles, fast_ions=())
    evolving_names = ('psi',)

    observed_calls = []
    original_calc = (
        inputs.models.neoclassical_model.calculate_conductivity_current_and_rotation
    )

    def spy_calc(runtime_params, geo, core_profiles, **kwargs):
      observed_calls.append((core_profiles, kwargs.get('fast_ions', ())))
      return original_calc(runtime_params, geo, core_profiles, **kwargs)

    with jax.disable_jit():
      with mock.patch.object(
          type(inputs.models.neoclassical_model),
          'calculate_conductivity_current_and_rotation',
          side_effect=spy_calc,
      ):
        calc_coeffs.calc_coeffs(
            runtime_params=inputs.runtime_params,
            geo=inputs.geo,
            solver_state=solver_state_lib.SolverState.from_core_profiles(
                stale_core_profiles
            ),
            explicit_model_outputs=inputs.explicit_model_outputs,
            models=inputs.models,
            evolving_names=evolving_names,
            model_state=inputs.model_state,
        )

    self.assertLen(observed_calls, 1)
    passed_core_profiles, passed_fast_ions = observed_calls[0]
    self.assertIsInstance(passed_core_profiles, solver_profiles.SolverProfiles)
    self.assertNotEmpty(passed_fast_ions)
    passed_he3 = [fi for fi in passed_fast_ions if fi.species == 'He3']
    self.assertLen(passed_he3, 1)
    np.testing.assert_allclose(passed_he3[0].n.value, 1e19)


if __name__ == '__main__':
  absltest.main()
