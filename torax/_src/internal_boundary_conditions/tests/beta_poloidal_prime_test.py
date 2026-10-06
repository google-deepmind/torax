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

"""Unit tests for the BetaPoloidalPrimeIBC model."""

import dataclasses
import typing
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import beta_poloidal_prime
from torax._src.internal_boundary_conditions import pydantic_model
from torax._src.orchestration import run_simulation
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class BetaPoloidalPrimeIBCTest(parameterized.TestCase):

  def test_defaults(self):
    ibc_config = pydantic_model.BetaPoloidalPrimeIBC(
        rho_norm_edge=0.85,
        n_e_edge=2.0e19,
        beta_poloidal_prime=1.5,
        Ti_Te_ratio=1.0,
    )
    self.assertFalse(ibc_config.n_e_is_fGW)
    self.assertIsNone(ibc_config.n_e_edge_multiplier)
    self.assertEqual(ibc_config.mode, beta_poloidal_prime.Mode.CONSTANT)
    self.assertIsNone(ibc_config.P_SOL_scaling)
    self.assertEqual(ibc_config.beta_poloidal_prime_min.get_value(0.0), 0.1)

  def test_missing_required_params_raises(self):
    with self.assertRaises(pydantic.ValidationError):
      pydantic_model.BetaPoloidalPrimeIBC.model_validate({})

  def test_missing_n_e_edge_and_multiplier_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Exactly one of 'n_e_edge' or 'n_e_edge_multiplier' must be provided",
    ):
      pydantic_model.BetaPoloidalPrimeIBC(
          rho_norm_edge=0.85,
          beta_poloidal_prime=1.5,
          Ti_Te_ratio=1.0,
      )

  def test_both_n_e_edge_and_multiplier_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Exactly one of 'n_e_edge' or 'n_e_edge_multiplier' must be provided",
    ):
      pydantic_model.BetaPoloidalPrimeIBC(
          rho_norm_edge=0.85,
          n_e_edge=2.0e19,
          n_e_edge_multiplier=1.5,
          beta_poloidal_prime=1.5,
          Ti_Te_ratio=1.0,
      )

  def test_power_dependent_without_p_sol_scaling_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError, 'P_SOL_scaling must be provided'
    ):
      pydantic_model.BetaPoloidalPrimeIBC(
          mode=beta_poloidal_prime.Mode.POWER_DEPENDENT,
          rho_norm_edge=0.85,
          n_e_edge=2.0e19,
          beta_poloidal_prime=1.5,
          Ti_Te_ratio=1.0,
      )

  def _setup_model_and_profiles(
      self,
      rho_norm_edge: float = 0.7,
      n_e_edge: float | None = 2.5e19,
      n_e_edge_multiplier: float | None = None,
      beta_poloidal_prime_val: float = 1.5,
      ti_te_ratio: float = 1.2,
      n_e_is_fGW: bool = False,
      Ip: float = 5e6,
      mode: beta_poloidal_prime.Mode = beta_poloidal_prime.Mode.CONSTANT,
      beta_poloidal_prime_min: float = 0.1,
      P_SOL_scaling: float | None = None,
      source_profiles: source_profiles_lib.SourceProfiles | None = None,
  ):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    core_profiles = dataclasses.replace(
        core_profiles,
        T_i=core_profile_helpers.make_constant_core_profile(geo, 1.0),
        T_e=core_profile_helpers.make_constant_core_profile(geo, 1.0),
        n_e=core_profile_helpers.make_constant_core_profile(geo, 3.0e19),
        n_i=core_profile_helpers.make_constant_core_profile(geo, 3.0e19),
        Ip_profile_face=jnp.linspace(0.0, Ip, geo.rho_face.shape[0]),
    )

    ibc_config = pydantic_model.BetaPoloidalPrimeIBC(
        mode=mode,
        rho_norm_edge=rho_norm_edge,
        n_e_edge=n_e_edge,
        n_e_edge_multiplier=n_e_edge_multiplier,
        beta_poloidal_prime=beta_poloidal_prime_val,
        beta_poloidal_prime_min=beta_poloidal_prime_min,
        P_SOL_scaling=P_SOL_scaling,
        Ti_Te_ratio=ti_te_ratio,
        n_e_is_fGW=n_e_is_fGW,
    )
    torax_pydantic.set_grid(ibc_config, geo.torax_mesh)
    ibc_rp = ibc_config.build_runtime_params(t=0.0)

    class _MockProfileConditions:
      internal_boundary_conditions = ibc_rp

    _MockProfileConditions.Ip = Ip

    class _MockRuntimeParams:
      profile_conditions = _MockProfileConditions()

    mock_runtime_params = typing.cast(
        runtime_params_lib.RuntimeParams, _MockRuntimeParams()
    )
    if source_profiles is None:
      source_profiles = mock.create_autospec(
          source_profiles_lib.SourceProfiles,
          instance=True,
          T_e={},
          T_i={},
      )
    model = beta_poloidal_prime.BetaPoloidalPrimeIBCModel()
    ibc_out = model(
        mock_runtime_params,
        geo,
        core_profiles,
        source_profiles=source_profiles,
    )
    return geo, core_profiles, ibc_out

  @parameterized.named_parameters(
      ('zero_p_sol_equals_min', 0.0, 0.2, 1.8, 10.0e6),
      ('intermediate_p_sol_tanh_scaling', 10.0e6, 0.2, 1.8, 10.0e6),
      ('high_p_sol_saturates_at_max', 100.0e6, 0.2, 1.8, 10.0e6),
  )
  def test_power_dependent_matches_expected_effective_beta_poloidal_prime(
      self,
      P_total: float,
      beta_poloidal_prime_min: float,
      beta_poloidal_prime_max: float,
      P_SOL_scaling: float,
  ):
    def _build_ibc_from_config(ibc_dict: dict[str, typing.Any]):
      config_dict = default_configs.get_default_config_dict()
      config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 25}
      config_dict['sources'] = {
          'generic_heat': {
              'P_total': P_total,
          }
      }
      config_dict['profile_conditions'][
          'internal_boundary_conditions'
      ] = ibc_dict
      torax_config = model_config.ToraxConfig.from_dict(config_dict)
      models = torax_config.build_models()
      sim_state, _, _ = run_simulation.prepare_simulation(torax_config)
      runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )(t=torax_config.numerics.t_initial)
      return models.internal_boundary_condition_model(
          runtime_params=runtime_params,
          geo=sim_state.geometry,
          core_profiles=sim_state.core_profiles,
          source_profiles=sim_state.core_sources,
      )

    ibc_power_dep = _build_ibc_from_config({
        'model_name': 'beta_poloidal_prime',
        'mode': 'power_dependent',
        'rho_norm_edge': 0.7,
        'n_e_edge': 2.5e19,
        'beta_poloidal_prime': beta_poloidal_prime_max,
        'beta_poloidal_prime_min': beta_poloidal_prime_min,
        'P_SOL_scaling': P_SOL_scaling,
        'Ti_Te_ratio': 1.2,
    })

    expected_beta_poloidal_prime = beta_poloidal_prime_min + (
        beta_poloidal_prime_max - beta_poloidal_prime_min
    ) * np.tanh(max(P_total, 0.0) / P_SOL_scaling)
    ibc_constant_equiv = _build_ibc_from_config({
        'model_name': 'beta_poloidal_prime',
        'mode': 'constant',
        'rho_norm_edge': 0.7,
        'n_e_edge': 2.5e19,
        'beta_poloidal_prime': expected_beta_poloidal_prime,
        'Ti_Te_ratio': 1.2,
    })

    np.testing.assert_allclose(ibc_power_dep.T_e, ibc_constant_equiv.T_e)
    np.testing.assert_allclose(ibc_power_dep.T_i, ibc_constant_equiv.T_i)
    np.testing.assert_allclose(ibc_power_dep.n_e, ibc_constant_equiv.n_e)

  def test_negative_p_sol_clamps_to_min(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    negative_sources = mock.create_autospec(
        source_profiles_lib.SourceProfiles,
        instance=True,
        T_e={'radiation': jnp.full_like(geo.rho_norm, -1.0e6)},
        T_i={},
    )
    _, _, ibc_power_dep = self._setup_model_and_profiles(
        mode=beta_poloidal_prime.Mode.POWER_DEPENDENT,
        beta_poloidal_prime_val=1.8,
        beta_poloidal_prime_min=0.2,
        P_SOL_scaling=10.0e6,
        source_profiles=negative_sources,
    )
    _, _, ibc_min_constant = self._setup_model_and_profiles(
        mode=beta_poloidal_prime.Mode.CONSTANT,
        beta_poloidal_prime_val=0.2,
    )
    np.testing.assert_allclose(ibc_power_dep.T_e, ibc_min_constant.T_e)
    np.testing.assert_allclose(ibc_power_dep.T_i, ibc_min_constant.T_i)

  def test_evaluates_profiles_in_edge_cells_only(self):
    geo, _, ibc_out = self._setup_model_and_profiles(rho_norm_edge=0.7)
    edge_cells = geo.rho_norm >= 0.7
    core_cells = geo.rho_norm < 0.7
    np.testing.assert_array_equal(ibc_out.T_e[core_cells], 0.0)
    np.testing.assert_array_equal(ibc_out.T_i[core_cells], 0.0)
    np.testing.assert_array_equal(ibc_out.n_e[core_cells], 0.0)
    self.assertTrue(np.all(ibc_out.T_e[edge_cells] > 0.0))
    self.assertTrue(np.all(ibc_out.T_i[edge_cells] > 0.0))
    self.assertTrue(np.all(ibc_out.n_e[edge_cells] > 0.0))

  def test_enforces_temperature_ratio(self):
    geo, _, ibc_out = self._setup_model_and_profiles(
        rho_norm_edge=0.7, ti_te_ratio=1.2
    )
    edge_cells = geo.rho_norm >= 0.7
    np.testing.assert_allclose(
        ibc_out.T_i[edge_cells] / ibc_out.T_e[edge_cells], 1.2
    )

  def test_two_point_face_mask_active_at_edge(self):
    geo, _, ibc_out = self._setup_model_and_profiles(rho_norm_edge=0.7)
    mask = ibc_out.get_two_point_face_mask(geo)
    self.assertTrue(np.any(mask))
    self.assertFalse(mask[0])

  def test_greenwald_density_conversion(self):
    fgw = 0.4
    Ip = 5e6
    geo, _, ibc_out_fgw = self._setup_model_and_profiles(
        rho_norm_edge=0.7,
        n_e_edge=fgw,
        n_e_is_fGW=True,
        Ip=Ip,
    )
    nGW = (Ip / 1e6) / (np.pi * geo.a_minor**2) * 1e20
    _, _, ibc_out_abs = self._setup_model_and_profiles(
        rho_norm_edge=0.7,
        n_e_edge=fgw * nGW,
        n_e_is_fGW=False,
        Ip=Ip,
    )
    np.testing.assert_allclose(ibc_out_fgw.n_e, ibc_out_abs.n_e)
    np.testing.assert_allclose(ibc_out_fgw.T_e, ibc_out_abs.T_e)
    np.testing.assert_allclose(ibc_out_fgw.T_i, ibc_out_abs.T_i)

  def test_n_e_edge_multiplier_matches_equivalent_n_e_edge(self):
    multiplier = 1.5
    _, core_profiles, ibc_out_mult = self._setup_model_and_profiles(
        rho_norm_edge=0.7,
        n_e_edge=None,
        n_e_edge_multiplier=multiplier,
    )
    n_e_sep = core_profiles.n_e.right_face_value.item()
    _, _, ibc_out_val = self._setup_model_and_profiles(
        rho_norm_edge=0.7,
        n_e_edge=multiplier * n_e_sep,
        n_e_edge_multiplier=None,
    )
    np.testing.assert_allclose(ibc_out_mult.n_e, ibc_out_val.n_e)
    np.testing.assert_allclose(ibc_out_mult.T_e, ibc_out_val.T_e)
    np.testing.assert_allclose(ibc_out_mult.T_i, ibc_out_val.T_i)

  @parameterized.named_parameters(
      ('value_m3', {'n_e_edge': 2.0e19, 'n_e_is_fGW': False}),
      ('value_fgw', {'n_e_edge': 0.3, 'n_e_is_fGW': True}),
      ('multiplier', {'n_e_edge_multiplier': 1.5}),
  )
  def test_solver_step_reconstructs_beta_poloidal_prime(
      self, n_e_edge_kwargs: dict[str, typing.Any]
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 25}
    config_dict['numerics']['evolve_density'] = True
    rho_norm_edge = 0.8
    target_beta_pol_prime = 1.5
    config_dict['profile_conditions']['internal_boundary_conditions'] = {
        'model_name': 'beta_poloidal_prime',
        'rho_norm_edge': rho_norm_edge,
        'beta_poloidal_prime': target_beta_pol_prime,
        'Ti_Te_ratio': 1.0,
        **n_e_edge_kwargs,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    sim_state, post_processed_outputs, step_fn = (
        run_simulation.prepare_simulation(torax_config)
    )
    output_state, post_processed_outputs = step_fn(
        sim_state, post_processed_outputs
    )
    geo = output_state.geometry

    # Faces strictly inside the edge region where the model is active
    edge_faces = (geo.rho_face_norm > rho_norm_edge) & (geo.rho_face_norm < 1.0)
    self.assertGreaterEqual(int(np.sum(edge_faces)), 2)

    np.testing.assert_allclose(
        post_processed_outputs.beta_pol_prime[edge_faces],
        target_beta_pol_prime,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
