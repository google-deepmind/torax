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

"""Unit tests for torax.torax_imastools.equilibrium.py"""

import dataclasses
import importlib
import os
from typing import Optional, Sequence
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import pytest
try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any
from torax.tests.test_lib import sim_test_case
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.config import build_sim
from torax import post_processing
from torax.torax_imastools.equilibrium import geometry_to_IMAS
from torax.torax_imastools.util import load_IMAS_data


class EquilibriumTest(sim_test_case.SimTestCase):
  """Unit tests for the `toraximastools.equilibrium` module."""
  @pytest.mark.skipif(
      importlib.util.find_spec('imaspy') is None,
      reason='IMASPy optional dependency'
    )
  @parameterized.parameters([
      dict(config_name='test_imas.py', rtol = 0.02, atol = 1e-8),
  ])
  def test_save_geometry_to_IMAS(self, config_name, rtol: Optional[float] = None, atol: Optional[float] = None,):
    """Test that the default IMAS geometry can be built and converted back to IDS."""
    if importlib.util.find_spec('imaspy') is None:
      self.skipTest('IMASPy optional dependency')

    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol
    #Input equilibrium reading
    config_module = self._get_config_module(config_name)
    geometry_dir = 'torax/data/third_party/geo'
    path = os.path.join(geometry_dir, config_module.CONFIG['geometry']['equilibrium_object'])
    equilibrium_in = load_IMAS_data(path, 'equilibrium')
    #Build TORAXSimState object and write output to equilibrium IDS.
    config_module.CONFIG['geometry']['n_rho'] = len(equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm) #Improve resolution to compare the input without losing to much information
    sim = build_sim.build_sim_from_config(config_module.CONFIG)
    #sim = self._get_sim(config_name)
    dynamic_runtime_params_slice, geo = (
        runtime_params_slice.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=sim.initial_state.t,
            dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
            geometry_provider=sim.geometry_provider,
        )
    )
    sim_state = post_processing.make_outputs(sim_state=sim.initial_state, geo=geo)
    sim_state.geometry = geo # To be removed once we track changes with main repository, as geometry is now part of sim_state.
    equilibrium_out = geometry_to_IMAS(sim_state)

    #Compare the output IDS with the input one.
    rhon_out = equilibrium_out.time_slice[0].profiles_1d.rho_tor_norm
    rhon_in = equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
    np.testing.assert_allclose(
      np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.psi),
      equilibrium_in.time_slice[0].profiles_1d.psi,
      rtol=rtol,
      atol=atol,
      err_msg = 'psi profile failed',
    )
    np.testing.assert_allclose(
      np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.q),
      equilibrium_in.time_slice[0].profiles_1d.q,
      rtol=rtol,
      atol=atol,
      err_msg = 'q profile failed',
    )
    np.testing.assert_allclose(
      np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.j_phi),
      equilibrium_in.time_slice[0].profiles_1d.j_phi,
      rtol=rtol,
      atol=atol,
      err_msg = 'jtot profile failed',
    )
    np.testing.assert_allclose(
      np.interp(rhon_in, rhon_out, equilibrium_out.time_slice[0].profiles_1d.f),
      equilibrium_in.time_slice[0].profiles_1d.f,
      rtol=rtol,
      atol=atol,
      err_msg = 'f profile failed',
    )


  @pytest.mark.skipif(
    importlib.util.find_spec('imaspy') is None,
    reason='IMASPy optional dependency'
  )
  @parameterized.parameters([
      dict(rtol = 0.02, atol = 1e-8),
  ])
  def test_geometry_from_IMAS(self, rtol: Optional[float] = None, atol: Optional[float] = None,):
    "Test to compare the outputs of CHEASE and IMAS methods for the same equilibrium."
    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol


    #Loading the equilibrium and constructing geometry object
    intermediate_IMAS = geometry.StandardGeometryIntermediates.from_IMAS(equilibrium_object = 'ITERhybrid_COCOS17_IDS_ddv4.nc', Ip_from_parameters = True)
    geo_IMAS = geometry.build_standard_geometry(intermediate_IMAS)

    intermediate_CHEASE = geometry.StandardGeometryIntermediates.from_chease()
    geo_CHEASE = geometry.build_standard_geometry(intermediate_CHEASE)

    #Comparison of the fields
    diverging_fields = []
    for key in geo_IMAS:
      if key != 'geometry_type' and key != 'Ip_from_parameters' and key != 'torax_mesh':
        try:
          np.testing.assert_allclose(geo_IMAS[key],
                                geo_CHEASE[key],
                                rtol = rtol,
                                atol = atol,
                                verbose = True,
                                err_msg= f'Value {key} failed'
                                )
        except AssertionError:
          diverging_fields.append(key)
    if diverging_fields != []:
      raise AssertionError(f'Diverging profiles: {diverging_fields}')

def face_to_cell(n_rho, face):
  cell = np.zeros(n_rho)
  cell[:] = 0.5 * (face[1:] + face[:-1])
  return cell


if __name__ == '__main__':
  absltest.main()
