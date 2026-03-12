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

from unittest import mock
from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib


class PedestalModelOutputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = mock.create_autospec(geometry.Geometry, instance=True)
    self.geo.rho_face_norm = jnp.linspace(0, 1, 10)
    self.geo.rho = jnp.linspace(0, 1, 9)
    self.pedestal_model_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.99,
        rho_norm_ped_top_idx=-1,
        T_i_ped=1.0,
        T_e_ped=1.1,
        n_e_ped=1.2e19,
        transport_multipliers=pedestal_model_output.TransportMultipliers(
            chi_e_multiplier=jnp.array(2.0),
            chi_i_multiplier=jnp.array(3.0),
            D_e_multiplier=jnp.array(4.0),
            v_e_multiplier=jnp.array(5.0),
        ),
    )

  def test_to_internal_boundary_conditions(self):
    ibc = self.pedestal_model_output.to_internal_boundary_conditions(self.geo)
    idx = self.pedestal_model_output.rho_norm_ped_top_idx
    with self.subTest('T_i'):
      np.testing.assert_allclose(
          ibc.T_i[idx], self.pedestal_model_output.T_i_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_i), self.pedestal_model_output.T_i_ped
      )
    with self.subTest('T_e'):
      np.testing.assert_allclose(
          ibc.T_e[idx], self.pedestal_model_output.T_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_e), self.pedestal_model_output.T_e_ped
      )
    with self.subTest('n_e'):
      np.testing.assert_allclose(
          ibc.n_e[idx], self.pedestal_model_output.n_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.n_e), self.pedestal_model_output.n_e_ped
      )

  def test_modify_core_transport_applies_multipliers(self):
    n_face = self.geo.rho_face_norm.shape[0]
    core_transport = state.CoreTransport(
        chi_face_ion=jnp.ones(n_face),
        chi_face_el=jnp.ones(n_face),
        d_face_el=jnp.ones(n_face),
        v_face_el=jnp.ones(n_face),
        chi_face_el_bohm=jnp.ones(n_face),
        chi_face_el_gyrobohm=jnp.ones(n_face),
        chi_face_ion_bohm=jnp.ones(n_face),
        chi_face_ion_gyrobohm=jnp.ones(n_face),
        chi_face_el_itg=jnp.ones(n_face),
        chi_face_el_tem=jnp.ones(n_face),
        chi_face_el_etg=jnp.ones(n_face),
        chi_face_ion_itg=jnp.ones(n_face),
        chi_face_ion_tem=jnp.ones(n_face),
        d_face_el_itg=jnp.ones(n_face),
        d_face_el_tem=jnp.ones(n_face),
        v_face_el_itg=jnp.ones(n_face),
        v_face_el_tem=jnp.ones(n_face),
        chi_neo_i=jnp.ones(n_face),
        chi_neo_e=jnp.ones(n_face),
        D_neo_e=jnp.ones(n_face),
        V_neo_e=jnp.ones(n_face),
        V_neo_ware_e=jnp.ones(n_face),
        chi_face_ion_pereverzev=jnp.ones(n_face),
        chi_face_el_pereverzev=jnp.ones(n_face),
        full_v_heat_face_ion_pereverzev=jnp.ones(n_face),
        full_v_heat_face_el_pereverzev=jnp.ones(n_face),
        d_face_el_pereverzev=jnp.ones(n_face),
        v_face_el_pereverzev=jnp.ones(n_face),
    )

    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.chi_max = jnp.array(1.0)
    pedestal_runtime_params.D_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_max = jnp.array(1.0)
    pedestal_runtime_params.V_e_min = jnp.array(-1.0)

    modified_core_transport = self.pedestal_model_output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )

    # Check turbulent and Pereverzev transport is scaled correctly.
    for field_name in [
        'chi_face_el',
        'chi_face_el_bohm',
        'chi_face_el_gyrobohm',
        'chi_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 2.0, 1.0),
      )
    for field_name in [
        'chi_face_ion',
        'chi_face_ion_bohm',
        'chi_face_ion_gyrobohm',
        'chi_face_ion_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 3.0, 1.0),
      )
    for field_name in [
        'd_face_el',
        'd_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 4.0, 1.0),
      )
    for field_name in [
        'v_face_el',
        'v_face_el_pereverzev',
    ]:
      field = getattr(modified_core_transport, field_name)
      np.testing.assert_allclose(
          field,
          jnp.where(pedestal_mask, 5.0, 1.0),
      )

    # Check neoclassical transport is not affected.
    np.testing.assert_allclose(
        modified_core_transport.chi_neo_i,
        core_transport.chi_neo_i,
    )
    np.testing.assert_allclose(
        modified_core_transport.chi_neo_e,
        core_transport.chi_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.D_neo_e,
        core_transport.D_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.V_neo_e,
        core_transport.V_neo_e,
    )
    np.testing.assert_allclose(
        modified_core_transport.V_neo_ware_e,
        core_transport.V_neo_ware_e,
    )


if __name__ == '__main__':
  absltest.main()
