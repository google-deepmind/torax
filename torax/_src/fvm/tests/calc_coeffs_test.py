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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src import model_state as model_state_lib
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.fvm import discrete_system
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config


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
    sources_config['fusion']['mode'] = 'ZERO'
    sources_config['ohmic']['mode'] = 'ZERO'
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(
                model_name='set_T_ped_n_ped',
                formation_model=dict(
                    model_name='prescribed',
                    pedestal_active=pedestal_active,
                ),
            ),
            sources=sources_config,
            solver=dict(
                use_predictor_corrector=False, theta_implicit=theta_implicit
            ),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(model_name='prescribed', chi_i=1)
                ),
                chi_min=0,
            ),
            time_step_calculator=dict(),
        )
    )
    models = torax_config.build_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=models.source_models,
        neoclassical_model=models.neoclassical_model,
    )
    evolving_names = tuple(['T_i'])
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        source_models=models.source_models,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=True,
    )
    model_state = model_state_lib.ModelState(
        pedestal=models.pedestal_model.initial_state(
            runtime_params, geo, core_profiles, explicit_source_profiles
        ),
        transport=models.transport_model.initial_state(runtime_params),
        sources=models.source_models.initial_state(runtime_params),
    )
    calc_coeffs.calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        models=models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
        pedestal_transition_state=pedestal_transition_state.PedestalTransitionState.empty(),
        model_state=model_state,
    )

  def test_calc_coeffs_hash(self):
    def create_coeffs_callback(
        torax_config: model_config.ToraxConfig,
    ) -> calc_coeffs.CoeffsCallback:
      models = torax_config.build_models()
      evolving_names = tuple(['T_i'])
      return calc_coeffs.CoeffsCallback(
          models=models,
          evolving_names=evolving_names,
      )

    torax_config = {
        'pedestal': {},
        'transport': {},
        'solver': {},
        'profile_conditions': {},
        'numerics': {},
        'sources': {},
        'geometry': {'geometry_type': 'circular'},
        'plasma_composition': {},
    }
    torax_config_with_pedestal = copy.deepcopy(torax_config)
    torax_config_with_pedestal['pedestal'] = {'model_name': 'set_T_ped_n_ped'}

    with self.subTest('same_coeffs_callback_hash_equal'):
      self.assertEqual(
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
      )
    with self.subTest('same_coeffs_callback_equal'):
      self.assertEqual(
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
      )
    with self.subTest('different_coeffs_callback_hash_not_equal'):
      self.assertNotEqual(
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config_with_pedestal)
              )
          ),
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
      )
    with self.subTest('different_coeffs_callback_not_equal'):
      self.assertNotEqual(
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config_with_pedestal)
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

    def _eval_rhs(cfg_dict):
      tcfg = model_config.ToraxConfig.from_dict(cfg_dict)
      models = tcfg.build_models()
      rp = build_runtime_params.RuntimeParamsProvider.from_config(tcfg)(
          t=tcfg.numerics.t_initial
      )
      geo = tcfg.geometry.build_provider(tcfg.numerics.t_initial)
      cp = initialization.initial_core_profiles(
          rp,
          geo,
          source_models=models.source_models,
          neoclassical_model=models.neoclassical_model,
      )
      if use_neumann_right_bc:
        cp = dataclasses.replace(
            cp,
            T_i=dataclasses.replace(
                cp.T_i,
                right_face_constraint=None,
                right_face_grad_constraint=jnp.array(-5.0),
            ),
            T_e=dataclasses.replace(
                cp.T_e,
                right_face_constraint=None,
                right_face_grad_constraint=jnp.array(-4.5),
            ),
            n_e=dataclasses.replace(
                cp.n_e,
                right_face_constraint=None,
                right_face_grad_constraint=jnp.array(-1.0e20),
            ),
        )
      explicit_src = source_profile_builders.build_source_profiles(
          runtime_params=rp,
          geo=geo,
          core_profiles=cp,
          source_models=models.source_models,
          explicit=True,
      )
      model_state = model_state_lib.ModelState(
          pedestal=models.pedestal_model.initial_state(
              rp, geo, cp, explicit_src
          ),
          transport=models.transport_model.initial_state(rp),
          sources=models.source_models.initial_state(rp),
      )
      evolving = ('T_i', 'T_e', 'n_e')
      coeffs = calc_coeffs.calc_coeffs(
          runtime_params=rp,
          geo=geo,
          core_profiles=cp,
          explicit_source_profiles=explicit_src,
          evolving_names=evolving,
          use_pereverzev=cfg_dict['solver']['use_pereverzev'],
          models=models,
          pedestal_transition_state=(
              pedestal_transition_state.PedestalTransitionState.empty()
          ),
          model_state=model_state,
      )
      x_vars = (cp.T_i, cp.T_e, cp.n_e)
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


if __name__ == '__main__':
  absltest.main()
