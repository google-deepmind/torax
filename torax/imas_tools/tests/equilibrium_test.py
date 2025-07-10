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

"""Unit tests for torax.imas_tools.equilibrium.py"""

import os
from typing import Optional
from unittest.mock import patch

from absl.testing import absltest
from absl.testing import parameterized
import imas
import numpy as np
import torax
from torax._src.geometry import geometry
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.geometry.pydantic_model import Geometry
from torax._src.geometry.pydantic_model import GeometryConfig
from torax._src.geometry.pydantic_model import IMASConfig
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config
from torax.imas_tools import equilibrium as imas_equilibrium
from torax.imas_tools import util as imas_util


class EquilibriumTest(sim_test_case.SimTestCase):
  """Unit tests for the `toraximastools.equilibrium` module."""

  @parameterized.parameters([
      dict(
          config_name='test_iterhybrid_predictor_corrector_imas.py',
          rtol=1.2e-1,
          atol=1e-8,
      ),
  ])
  def test_save_geometry_to_IMAS(
      self,
      config_name,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Test that the default IMAS geometry can be built and converted back
    to IDS."""
    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol
    # Input equilibrium reading
    config_module = self._get_config_dict(config_name)
    geometry_directory = 'torax/data/third_party/geo'
    path = os.path.join(
        geometry_directory, config_module['geometry']['imas_filepath']
    )
    equilibrium_in = imas_util.load_IMAS_data(path, 'equilibrium')
    # Build TORAXSimState object and write output to equilibrium IDS.
    # Improve resolution to compare the input without losing too much
    # information
    config_module['geometry']['n_rho'] = len(
        equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
    )
    torax_config = model_config.ToraxConfig.from_dict(config_module)

    (
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        initial_state,
        initial_post_processed_outputs,
        _,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    sim_state = initial_state
    torax_config_dict = get_geometry_config_dict(torax_config)
    config_kwargs = {
        **torax_config_dict,
        'equilibrium_object': equilibrium_in,
        'imas_filepath': None,
    }
    imas_cfg = IMASConfig(**config_kwargs)
    cfg = GeometryConfig(config=imas_cfg)
    step_fn._geometry_provider = Geometry(
        geometry_type=geometry.GeometryType.IMAS,
        geometry_configs={str(equilibrium_in.time[0]): cfg},
    ).build_provider

    sim_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            t=torax_config.numerics.t_initial,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=step_fn._geometry_provider,
            step_fn=step_fn,
        )
    )

    equilibrium_out = imas_equilibrium.geometry_to_IMAS(
        sim_state,
        post_processed_outputs,
    )

    rhon_out = equilibrium_out.time_slice[0].profiles_1d.rho_tor_norm
    rhon_in = equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
    for attr1, attr2 in [
        ('profiles_1d', 'phi'),
        ('profiles_1d', 'psi'),
        ('profiles_1d', 'q'),
        ('profiles_1d', 'gm2'),
        # j_phi goes through too many derivatives and calculations
        # for proper test probably
        # ('profiles_1d', 'j_phi'),
    ]:
      # Compare the output IDS with the input one.
      var_in = getattr(getattr(equilibrium_in.time_slice[0], attr1), attr2)
      var_out = getattr(getattr(equilibrium_out.time_slice[0], attr1), attr2)
      n = int(var_in.size / 10)
      np.testing.assert_allclose(
          np.interp(rhon_in, rhon_out, var_out)[n:-n],
          var_in[n:-n],
          rtol=rtol,
          atol=atol,
          err_msg=f'{attr1} {attr2} failed',
      )

  @parameterized.parameters([dict(rtol=1e-1, atol=1e-8)])
  def test_geometry_from_IMAS(
      self,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Test to compare the outputs of CHEASE and IMAS methods for the same
    equilibrium."""
    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol

    # Loading the equilibrium and constructing geometry object
    config = geometry_pydantic_model.IMASConfig(
        imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc',
        Ip_from_parameters=True,
    )
    geo_IMAS = config.build_geometry()

    geo_CHEASE = geometry_pydantic_model.CheaseConfig().build_geometry()

    # Comparison of the fields
    diverging_fields = []
    for key in geo_IMAS.__dict__.keys():
      if (
          key != 'geometry_type'
          and key != 'Ip_from_parameters'
          and key != 'torax_mesh'
          and key != '_z_magnetic_axis'
      ):
        try:
          a = geo_IMAS.__dict__[key]
          b = geo_CHEASE.__dict__[key]
          if a.size > 8:
            a = a[4:-4]
            b = b[4:-4]
          np.testing.assert_allclose(
              a,
              b,
              rtol=rtol,
              atol=atol,
              verbose=True,
              err_msg=f'Value {key} failed',
          )
        except AssertionError:
          diverging_fields.append(key)
    if diverging_fields:
      raise AssertionError(f'Diverging profiles: {diverging_fields}')

  @parameterized.parameters([
      dict(input_mode='imas_filepath'),
      dict(input_mode='imas_uri'),
      dict(input_mode='equilibrium_object'),
  ])
  def test_IMAS_input(self, input_mode: str):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    full_path = f'{torax.__path__[0]}/data/third_party/geo/{filename}'
    kwargs = {}
    if input_mode == 'imas_filepath':
      kwargs['imas_filepath'] = full_path
      config = geometry_pydantic_model.IMASConfig(**kwargs)
      config.build_geometry()
    elif input_mode == 'imas_uri':
      # imas_core not available for CI so just check if loader is called
      with patch('torax.imas_tools.util.DBEntry') as mocked_class:
        mocked_class.return_value = imas.DBEntry(uri=full_path, mode='r')
        kwargs['imas_uri'] = f'imas:hdf5?path={full_path}'
        config = geometry_pydantic_model.IMASConfig(**kwargs)
        config.build_geometry()
    elif input_mode == 'equilibrium_object':
      equilibrium_in = imas_util.load_IMAS_data(full_path, 'equilibrium')
      kwargs['equilibrium_object'] = equilibrium_in
      config = geometry_pydantic_model.IMASConfig(**kwargs)
      config.build_geometry()
    else:
      raise ValueError('input_mode should be one of the viable modes.')


def get_geometry_config_dict(config: model_config.ToraxConfig) -> dict:
  """
  Obtain config dict for IMAS geometry based on existing one.
  """
  # only get overlapping keys from given config and IMASConfig
  imas_config_keys = IMASConfig.__annotations__
  # we can pick a random entry since all fields are time_invariant except
  # hires_fac which we can ignore and equilibrium_object which we overwrite
  if isinstance(config.geometry.geometry_configs, dict):
    config_dict = list(config.geometry.geometry_configs.values())[
        0
    ].config.__dict__
  else:
    config_dict = config.geometry.geometry_configs.config.__dict__
  config_dict = {
      key: value
      for key, value in config_dict.items()
      if key in imas_config_keys
  }
  config_dict['geometry_type'] = 'imas'
  return config_dict


if __name__ == '__main__':
  absltest.main()
