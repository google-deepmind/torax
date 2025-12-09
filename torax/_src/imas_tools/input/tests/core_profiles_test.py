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
from absl.testing import absltest
import numpy as np
from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class CoreProfilesTest(sim_test_case.SimTestCase):

  def test_offset_time(self):
    offset = 100.0
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    ids_in = loader.load_imas_data(path, "core_profiles")
    core_profiles_conditions = core_profiles.profile_conditions_from_IMAS(
        ids_in,
        t_initial=offset,
    )
    t_in = np.array([
        float(ids_in.profiles_1d[i].time)
        for i in range(len(ids_in.profiles_1d))
    ])
    t_out = np.array(list(core_profiles_conditions["Ip"][0]))
    np.testing.assert_equal(t_out, t_in + 100.0)

  def test_init_profiles_from_IMAS(self):
    """Test to compare initialized profiles for consistency.

    The IMAS netCDF file comes from a METIS ITER baseline scenario time
    slice taken on the flat top. It is useful to have such file with
    profiles having more radial resolution. The IDS grid is made of 21
    points.
    """
    rtol = 1e-6
    atol = 1e-8
    # Input core_profiles reading and config loading
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")
    path = "core_profiles_15MA_DT_50_50_flat_top_slice.nc"
    ids_in = loader.load_imas_data(path, "core_profiles")
    rhon_in = ids_in.profiles_1d[0].grid.rho_tor_norm

    # Modifying the input config profiles_conditions class
    core_profiles_conditions = core_profiles.profile_conditions_from_IMAS(
        ids_in,
        t_initial=0.0,
    )
    config["geometry"]["n_rho"] = 200
    config["profile_conditions"] = core_profiles_conditions
    torax_config = model_config.ToraxConfig.from_dict(config)

    # Init sim from config
    sim_state, _, _ = run_simulation.prepare_simulation(torax_config)

    # Read output values
    torax_mesh = torax_config.geometry.build_provider.torax_mesh
    cell_centers = torax_mesh.cell_centers
    # Compare the initial core_profiles with the ids profiles
    init_core_profiles = sim_state.core_profiles
    np.testing.assert_allclose(
        init_core_profiles.T_e.value * 1e3,
        np.interp(
            cell_centers, rhon_in, ids_in.profiles_1d[0].electrons.temperature,
        ),
        rtol=rtol,
        atol=atol,
        err_msg="Te profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.T_i.value * 1e3,
        np.interp(
            cell_centers, rhon_in, ids_in.profiles_1d[0].t_i_average
        ),
        rtol=rtol,
        atol=atol,
        err_msg="Ti profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.n_e.value,
        np.interp(
            cell_centers, rhon_in, ids_in.profiles_1d[0].electrons.density,
        ),
        rtol=rtol,
        atol=atol,
        err_msg="ne profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.psi.value,
        np.interp(
            cell_centers, rhon_in, ids_in.profiles_1d[0].grid.psi
        ),
        rtol=rtol,
        atol=atol,
        err_msg="psi profile failed",
    )

  def test_imas_plasma_composition(self):
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")

    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    core_profiles_in = loader.load_imas_data(path, "core_profiles")
    # Indivual ion info empty in the inital IDS so create fake ions data
    core_profiles_in.profiles_1d[0].ion.resize(3)
    core_profiles_in.profiles_1d[1].ion.resize(3)
    core_profiles_in.global_quantities.ion.resize(3)
    core_profiles_in.profiles_1d[0].ion[0].name = "D"
    core_profiles_in.profiles_1d[0].ion[0].density = [9e19, 3e19]
    core_profiles_in.profiles_1d[1].ion[0].density = [9e19, 3e19]
    core_profiles_in.profiles_1d[0].ion[1].name = "T"
    core_profiles_in.profiles_1d[0].ion[1].density = [9e19, 3e19]
    core_profiles_in.profiles_1d[1].ion[1].density = [9e19, 3e19]
    core_profiles_in.profiles_1d[0].ion[2].name = "Ne"
    core_profiles_in.profiles_1d[1].ion[2].name = "Ne"
    core_profiles_in.profiles_1d[0].ion[2].density = (
        core_profiles_in.profiles_1d[0].electrons.density / 100
    )
    core_profiles_in.profiles_1d[1].ion[2].density = (
        core_profiles_in.profiles_1d[1].electrons.density / 100
    )

    with self.subTest(name="Missing expected impurity."):
      self.assertRaises(
          ValueError,
          core_profiles.plasma_composition_from_IMAS,
          core_profiles_in,
          None,
          None,
          ["Xe"],
      )
    with self.subTest(name="Missing expected main ion."):
      self.assertRaises(
          ValueError,
          core_profiles.plasma_composition_from_IMAS,
          core_profiles_in,
          None,
          ["H"],
          None,
      )
    with self.subTest(name="Invalid ion name."):
      self.assertRaises(
          KeyError,
          core_profiles.plasma_composition_from_IMAS,
          core_profiles_in,
          None,
          None,
          ["He5"],
      )
    with self.subTest(name="Test config is properly built."):
      plasma_composition_data = core_profiles.plasma_composition_from_IMAS(
          core_profiles_in,
          t_initial=None,
          main_ions_symbols=["D", "T"],
          expected_impurities=["Ne"],
      )
      config["plasma_composition"] = plasma_composition_data
      config["plasma_composition"]["impurity"]["impurity_mode"] = "n_e_ratios"
      torax_config = model_config.ToraxConfig.from_dict(config)

      Ne_impurity = torax_config.plasma_composition.impurity.species["Ne"]
      self.assertIsNotNone(Ne_impurity)
      np.testing.assert_equal(
          torax_config.plasma_composition.get_main_ion_names(), ("D", "T")
      )
      np.testing.assert_equal(
          torax_config.plasma_composition.get_impurity_names(), ("Ne",)
      )
      np.testing.assert_allclose(Ne_impurity.get_value(0.0), 0.01)
      np.testing.assert_allclose(
          torax_config.plasma_composition.main_ion["D"].get_value(0.0), 0.5
      )


if __name__ == "__main__":
  absltest.main()
