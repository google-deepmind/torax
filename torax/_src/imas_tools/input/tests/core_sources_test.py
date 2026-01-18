# Copyright 2025 DeepMind Technologies Limited
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
import pathlib

from absl.testing import absltest
import numpy as np
from torax._src.imas_tools.input import core_sources
from torax._src.imas_tools.input import loader
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class CoreSourcesTest(sim_test_case.SimTestCase):

  def test_combine_same_type_sources(self):
    """Tests the addition method of SourceCollection when receiving an already existing source."""
    # Generates dummy profiles for 2 ec sources.
    time = np.array([0.0, 1.0])
    rho_norm = np.array([0.0, 0.5, 1.0])
    ec1_heat = np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
    ec1_curr = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    ec2_heat = np.array([[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]])
    ec2_curr = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    ec1 = core_sources._SourceProfiles(
        time=time, rho_norm=rho_norm, T_e=ec1_heat, psi=ec1_curr
    )
    ec2 = core_sources._SourceProfiles(
        time=time,
        rho_norm=rho_norm,
        T_e=ec2_heat,
        psi=ec2_curr,
    )
    collection = core_sources._SourceCollection()
    collection.add("ecrh", ec1)
    # Adding another ecrh source should trigger the combine method.
    collection.add("ecrh", ec2)

    # Checks the output is properly built.
    result_dict = collection.to_dict()
    ecrh_data = result_dict["ecrh"]["prescribed_values"]
    assert "ecrh" in result_dict
    assert len(ecrh_data) == 2
    total_heat = ecrh_data[0][2]
    expected_heat = ec1_heat + ec2_heat
    np.testing.assert_allclose(total_heat, expected_heat)
    total_current = ecrh_data[1][2]
    expected_current = ec1_curr + ec2_curr
    np.testing.assert_allclose(total_current, expected_current)

  def test_sources_from_IMAS(self):
    """Test to compare initialized profiles for consistency.

    The IMAS netCDF file comes from the same METIS ITER baseline scenario as
    core_profiles_15MA_DT_50_50_flat_top_slice.nc in data/third_party/imas, and
    was converted to DD version 4.0.0
    """
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")
    directory = pathlib.Path(__file__).parent
    path = "core_sources_ddv4.nc"
    ids_name = "core_sources"
    ids_in = loader.load_imas_data(path, ids_name, directory=directory)
    sources = core_sources.sources_from_IMAS(ids_in, None, False)
    config["sources"] |= sources
    # Checks the config can be built properly with input sources.
    torax_config = model_config.ToraxConfig.from_dict(config)


if __name__ == "__main__":
  absltest.main()
