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
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import beta_poloidal_prime
from torax._src.internal_boundary_conditions import pydantic_model
from torax._src.orchestration import run_simulation
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

  def test_missing_required_params_raises(self):
    with self.assertRaises(pydantic.ValidationError):
      pydantic_model.BetaPoloidalPrimeIBC.model_validate({})

  def _setup_model_and_profiles(
      self,
      rho_norm_edge: float = 0.7,
      n_e_edge: float = 2.5e19,
      beta_poloidal_prime_val: float = 1.5,
      ti_te_ratio: float = 1.2,
      n_e_is_fGW: bool = False,
      Ip: float = 5e6,
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
        rho_norm_edge=rho_norm_edge,
        n_e_edge=n_e_edge,
        beta_poloidal_prime=beta_poloidal_prime_val,
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
    model = beta_poloidal_prime.BetaPoloidalPrimeIBCModel()
    ibc_out = model(mock_runtime_params, geo, core_profiles)
    return geo, core_profiles, ibc_out

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

  @parameterized.parameters(False, True)
  def test_solver_step_reconstructs_beta_poloidal_prime(self, n_e_is_fGW: bool):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 25}
    config_dict['numerics']['evolve_density'] = True
    rho_norm_edge = 0.8
    target_beta_pol_prime = 1.5
    n_e_edge = 0.3 if n_e_is_fGW else 2.0e19
    config_dict['profile_conditions']['internal_boundary_conditions'] = {
        'model_name': 'beta_poloidal_prime',
        'rho_norm_edge': rho_norm_edge,
        'n_e_edge': n_e_edge,
        'n_e_is_fGW': n_e_is_fGW,
        'beta_poloidal_prime': target_beta_pol_prime,
        'Ti_Te_ratio': 1.0,
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
