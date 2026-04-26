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
from unittest import mock

from absl.testing import absltest
import imas
import numpy as np
import torax
from torax._src.imas_tools.input import core_sources
from torax._src.imas_tools.input import loader
from torax._src.sources import source as source_module
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

    ecrh_affected_profiles = (
        source_module.AffectedCoreProfile.TEMP_EL,
        source_module.AffectedCoreProfile.PSI,
    )
    profiles1 = {
        source_module.AffectedCoreProfile.TEMP_EL: ec1_heat,
        source_module.AffectedCoreProfile.PSI: ec1_curr,
    }
    profiles2 = {
        source_module.AffectedCoreProfile.TEMP_EL: ec2_heat,
        source_module.AffectedCoreProfile.PSI: ec2_curr,
    }
    ec1 = core_sources._SourceProfiles(
        time=time,
        rho_norm=rho_norm,
        affected_profiles=ecrh_affected_profiles,
        profiles=profiles1,
    )
    ec2 = core_sources._SourceProfiles(
        time=time,
        rho_norm=rho_norm,
        affected_profiles=ecrh_affected_profiles,
        profiles=profiles2,
    )
    collection = core_sources._SourceCollection()
    collection.add("ecrh", ec1)
    # Adding another ecrh source should trigger the combine method.
    collection.add("ecrh", ec2)

    # Checks the output is properly built.
    result_dict = collection.to_dict()
    ecrh_data = result_dict["ecrh"]["prescribed_values"]
    assert len(ecrh_data) == 2
    total_heat = ecrh_data[0][2]
    expected_heat = ec1_heat + ec2_heat
    np.testing.assert_allclose(total_heat, expected_heat)
    total_current = ecrh_data[1][2]
    expected_current = ec1_curr + ec2_curr
    np.testing.assert_allclose(total_current, expected_current)

  def test_sources_from_IMAS(self):
    """Test to check config can be built from IMAS sources.

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
    _, imas_results = torax.run_simulation(torax_config, progress_bar=False)

  def test_get_particle_profile(self):
    """Tests that particle profile is extracted from electrons when available."""
    core_source = imas.IDSFactory().core_sources()
    core_source.source.resize(3)
    core_source.source[0].profiles_1d.resize(1)
    core_source.source[1].profiles_1d.resize(1)
    core_source.source[2].profiles_1d.resize(1)
    core_source.source[0].profiles_1d[0].electrons.particles = [1.0, 2.0]
    core_source.source[1].profiles_1d[0].ion.resize(1)
    core_source.source[1].profiles_1d[0].ion[0].element.resize(1)
    core_source.source[1].profiles_1d[0].ion[0].particles = [1.0, 2.0]
    core_source.source[1].profiles_1d[0].ion[0].element[0].z_n = 1.0
    with self.subTest(name="electrons"):
      res = core_sources._get_particle_profile(
          core_source.source[0].profiles_1d
      )
      np.testing.assert_allclose(res[0], [1.0, 2.0])
    with self.subTest(name="ions"):
      res = core_sources._get_particle_profile(
          core_source.source[1].profiles_1d
      )
      np.testing.assert_allclose(res[0], [1.0, 2.0])
    with self.subTest(name="Missing profiles"):
      with self.assertRaisesRegex(ValueError, "Expected particle source"):
        core_sources._get_particle_profile(core_source.source[2].profiles_1d)


if __name__ == "__main__":
  absltest.main()
