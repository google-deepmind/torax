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
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import imas
import numpy as np
from torax._src.geometry import geometry_loader
from torax._src.imas_tools import equilibrium as imas_geometry
from torax._src.geometry import pydantic_model as geometry_pydantic_model


# pylint: disable=invalid-name
class EquilibriumTest(parameterized.TestCase):

  @parameterized.parameters([dict(rtol=1e-1, atol=1e-8)])
  def test_geometry_from_IMAS(
      self,
      rtol: float | None = None,
      atol: float | None = None,
  ):
    """Test to compare CHEASE and IMAS methods for the same equilibrium."""
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

  def test_IMAS_input_with_filepath(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    config = geometry_pydantic_model.IMASConfig(imas_filepath=filename)
    config.build_geometry()

  def test_IMAS_input_with_uri(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    geometry_directory = geometry_loader.get_geometry_dir()
    full_path = os.path.join(geometry_directory, filename)
    mock_value = imas.DBEntry(uri=full_path, mode='r')
    # imas_core not available for CI so just check if loader is called
    with mock.patch('imas.DBEntry') as mocked_class:
      mocked_class.return_value = mock_value
      imas_uri = f'imas:hdf5?path={full_path}'
      config = geometry_pydantic_model.IMASConfig(
          imas_uri=imas_uri, imas_filepath=None
      )
      config.build_geometry()

  def test_IMAS_input_with_equilibrium_object(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    equilibrium_in = imas_geometry._load_geo_data(filename)
    config = geometry_pydantic_model.IMASConfig(
        equilibrium_object=equilibrium_in, imas_filepath=None
    )
    config.build_geometry()


if __name__ == '__main__':
  absltest.main()
