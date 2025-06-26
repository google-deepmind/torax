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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from torax._src import constants
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.core_profiles import updaters
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class BoundaryConditionsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          n_e={0.0: {0.0: 1.5e20, 1.0: 1.0e20}},
          n_e_right_bc=None,
          expected_n_e_right_bc=1.0e20,  # Value from profile.
      ),
      dict(
          n_e={0.0: {0.0: 1.5e20, 1.0: 1.0e20}},
          n_e_right_bc=(
              (np.array([0.0, 0.1]), np.array([0.1e20, 2.0e20])),
              'step',
          ),
          expected_n_e_right_bc=2.0e20,  # Value from boundary condition.
      ),
  )
  def test_setting_boundary_conditions(
      self,
      n_e_right_bc,
      n_e,
      expected_n_e_right_bc,
  ):
    """Tests that setting boundary conditions works."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # state, we want to grab the boundary condition params at time 0.
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'T_i': {0.0: {0.0: 27.7, 1.0: 1.0}},
        'T_e': {0.0: {0.0: 42.0, 1.0: 0.1}, 1.0: 0.1},
        'T_i_right_bc': 27.7,
        'T_e_right_bc': {0.0: 42.0, 1.0: 0.1},
        'n_e_right_bc': n_e_right_bc,
        'n_e': n_e,
        'n_e_nbar_is_fGW': False,
        'Ip': {0.0: 5e6, 1.0: 7e6},
        'normalize_n_e_to_nbar': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    initial_dynamic_runtime_params_slice = provider(
        t=torax_config.numerics.t_initial
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        initial_dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    dynamic_runtime_params_slice = provider(t=0.5)

    bc = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,  # Not used
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        core_profiles_t=core_profiles,
        static_runtime_params_slice=static_slice,
        geo_t_plus_dt=geo,
    )
    # pylint: disable=invalid-name
    T_i = dataclasses.replace(
        core_profiles.T_i,
        **bc['T_i'],
    )
    T_e = dataclasses.replace(
        core_profiles.T_e,
        **bc['T_e'],
    )
    psi = dataclasses.replace(core_profiles.psi, **bc['psi'])
    n_e = dataclasses.replace(
        core_profiles.n_e,
        **bc['n_e'],
    )
    n_i = dataclasses.replace(
        core_profiles.n_i,
        **bc['n_i'],
    )
    n_impurity = dataclasses.replace(
        core_profiles.n_impurity,
        **bc['n_impurity'],
    )
    updated = dataclasses.replace(
        core_profiles,
        T_e=T_e,
        T_i=T_i,
        n_e=n_e,
        n_i=n_i,
        n_impurity=n_impurity,
        psi=psi,
    )

    psi_constraint = (
        6e6
        * (16 * np.pi**3 * constants.CONSTANTS.mu0 * geo.Phi_b)
        / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
    )
    # pylint: disable=invalid-name
    Z_i_face = core_profiles.Z_i_face
    Z_eff_face = dynamic_runtime_params_slice.plasma_composition.Z_eff_face
    Z_impurity_face = core_profiles.Z_impurity_face
    # pylint: enable=invalid-name
    dilution_factor_face = (Z_impurity_face - Z_eff_face) / (
        Z_i_face * (Z_impurity_face - Z_i_face)
    )
    expected_n_i_bound_right = expected_n_e_right_bc * dilution_factor_face[-1]
    expected_n_impurity_bound_right = (
        expected_n_e_right_bc - expected_n_i_bound_right * Z_i_face[-1]
    ) / Z_impurity_face[-1]
    np.testing.assert_allclose(updated.T_i.right_face_constraint, 27.7)
    np.testing.assert_allclose(updated.T_e.right_face_constraint, 21.05)
    np.testing.assert_allclose(
        updated.n_e.right_face_constraint,
        expected_n_e_right_bc,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        updated.n_i.right_face_constraint, expected_n_i_bound_right
    )
    np.testing.assert_allclose(
        updated.n_impurity.right_face_constraint,
        expected_n_impurity_bound_right,
    )
    np.testing.assert_allclose(updated.T_e.right_face_constraint, 21.05)
    np.testing.assert_allclose(
        updated.psi.right_face_grad_constraint, psi_constraint
    )


if __name__ == '__main__':
  absltest.main()
