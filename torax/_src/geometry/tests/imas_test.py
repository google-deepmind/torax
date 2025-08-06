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
from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import imas
import numpy as np
import pydantic
from torax._src.geometry import geometry_loader
from torax._src.geometry import imas as imas_geometry
from torax._src.geometry import pydantic_model as geometry_pydantic_model


# pylint: disable=invalid-name
class EquilibriumTest(parameterized.TestCase):

  @parameterized.parameters([dict(rtol=1e-1, atol=1e-8)])
  def test_geometry_from_IMAS(
      self,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
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
    equilibrium_in = imas_geometry._load_imas_data(filename)
    config = geometry_pydantic_model.IMASConfig(
        equilibrium_object=equilibrium_in, imas_filepath=None
    )
    config.build_geometry()

  def test_IMAS_loading_specific_slice(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    SLICE_TIME = 5
    SLICE_INDEX = 1
    config_at_0 = geometry_pydantic_model.IMASConfig(imas_filepath=filename)
    config_at_slice_from_time = geometry_pydantic_model.IMASConfig(
        imas_filepath=filename, slice_time=SLICE_TIME
    )
    config_at_slice_from_index = geometry_pydantic_model.IMASConfig(
        imas_filepath=filename, slice_index=SLICE_INDEX
    )
    geo_at_0 = config_at_0.build_geometry()
    geo_at_slice_from_time = config_at_slice_from_time.build_geometry()
    geo_at_slice_from_index = config_at_slice_from_index.build_geometry()

    for key in geo_at_0.__dict__.keys():
      np.testing.assert_allclose(
          geo_at_slice_from_time[key],
          geo_at_slice_from_index[key],
          err_msg=f'Value {key} mismatched between slice_time and slice_index',
      )
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(
            geo_at_0[key],
            geo_at_slice_from_time[key],
        )
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(
            geo_at_0[key],
            geo_at_slice_from_index[key],
        )

  def test_IMAS_raises_if_slice_specified_twice(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'exactly one of `slice_time` and `slice_index`',
    ):
      geometry_pydantic_model.IMASConfig(
          imas_filepath=filename, slice_index=1, slice_time=1
      )

  def test_IMAS_raises_if_slice_out_of_range(self):
    filename = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
    with self.assertRaisesRegex(
        IndexError,
        'out of range',
    ):
      config = geometry_pydantic_model.IMASConfig(
          imas_filepath=filename,
          slice_index=100,
      )
      config.build_geometry()


if __name__ == '__main__':
  absltest.main()
