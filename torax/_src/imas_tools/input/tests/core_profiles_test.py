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


import os
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torax
from torax._src.imas_tools.input.core_profiles import core_profiles_from_IMAS
from torax._src.imas_tools.input.core_profiles import update_dict
from torax._src.imas_tools.input.loader import load_imas_data
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


class CoreProfilesTest(sim_test_case.SimTestCase):
  """Unit tests for torax.torax_imastools.input.core_profiles.py"""

  def test_update_dict(
      self,
  ):
    """Unit tests for the update_dict method."""
    old_dict = {
        "str_key": 3,
        "nested": {"x1": 5, "x2": 3},
        "profiles": {0.0: "old", 1.0: "old"},
    }
    simple_update = {"str_key": 5}
    nested_update = {"nested": {"x1": 0, "x2": 1}}
    profiles_update = {"profiles": {0.0: "new", 1.0: "new"}}
    with self.subTest("Test simple update of a str key."):
      new_dict = update_dict(old_dict, simple_update)
      assert new_dict["str_key"] == simple_update["str_key"]
      assert new_dict["str_key"] is not old_dict["str_key"]
    with self.subTest("Test update of a nested dict with str keys."):
      new_dict = update_dict(old_dict, nested_update)
      assert new_dict["nested"] == nested_update["nested"]
      assert new_dict["nested"] is not old_dict["nested"]
    with self.subTest(
        "Test simple update of a profiles type dict with floats as keys."
    ):
      new_dict = update_dict(old_dict, profiles_update)
      assert new_dict["profiles"] == profiles_update["profiles"]
      assert new_dict["profiles"] is not old_dict["profiles"]

  def test_offset_time(
      self,
  ):
    """Unit tests to check the t_initial optional args offset correctly the
    time array."""
    offset = 100.0
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    ids_in = load_imas_data(path, "core_profiles", dir)
    core_profiles_conditions = core_profiles_from_IMAS(
        ids_in,
        t_initial=offset,
    )
    t_in = np.array([
        float(ids_in.profiles_1d[i].time)
        for i in range(len(ids_in.profiles_1d))
    ])
    t_out = np.array(
        list(core_profiles_conditions["profile_conditions"]["Ip"][0])
    )
    np.testing.assert_equal(t_out, t_in + 100.0)

  def test_run_with_core_profiles_to_IMAS(
      self,
  ):
    """Test that TORAX simulation example can run with input core_profiles
    ids profiles, without raising error.

    The IMAS netCDF file was generated manually filling the core_profiles
    IDS based on the profile_conditions contained in the
    iterhybrid_rampup.py example config (after unit conversions).
    """
    # Input core_profiles reading and config loading
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")

    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    core_profiles_in = load_imas_data(path, "core_profiles", dir)

    # Modifying the input config profiles_conditions class
    core_profiles_data = core_profiles_from_IMAS(
        core_profiles_in,
    )
    imas_profile_conditions = {
        "profile_conditions": {
            **core_profiles_data["profile_conditions"],
        },
    }
    config = update_dict(config, imas_profile_conditions)

    torax_config = model_config.ToraxConfig.from_dict(config)

    # Run Sim
    _, results = torax.run_simulation(torax_config)
    # Check that the simulation completed successfully.
    if results.sim_error != torax.SimError.NO_ERROR:
      raise ValueError(
          f"TORAX failed to run the simulation with error: {results.sim_error}."
      )

  @parameterized.parameters([
      dict(config_name="test_iterhybrid_rampup_short.py", rtol=1e-6, atol=1e-8),
  ])
  def test_init_profiles_from_IMAS(
      self,
      config_name,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Test to compare initialized profiles in TORAX with the initial
    core_profiles used to check consistency.

    The IMAS netCDF file comes from a METIS ITER baseline scenario time
    slice taken on the flat top. It is useful to have such file with
    profiles having more radial resolution. The IDS grid is made of 21
    points.
    """
    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol
    # Input core_profiles reading and config loading
    config = self._get_config_dict(config_name)
    path = (  # Using this as input instead of rampup_conditions because it has more radial resolution.
        "core_profiles_15MA_DT_50_50_flat_top_slice.nc"
    )
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    core_profiles_in = load_imas_data(path, "core_profiles", dir)
    rhon_in = core_profiles_in.profiles_1d[0].grid.rho_tor_norm

    # Modifying the input config profiles_conditions class
    core_profiles_data = core_profiles_from_IMAS(
        core_profiles_in,
        t_initial=0.0,
    )
    imas_profile_conditions = {
        "profile_conditions": {
            **core_profiles_data["profile_conditions"],
        },
    }
    config["geometry"]["n_rho"] = 200
    config = update_dict(config, imas_profile_conditions)
    torax_config = model_config.ToraxConfig.from_dict(config)

    # Init sim from config
    _, sim_state, _, _ = prepare_simulation(torax_config)

    # Read output values
    torax_mesh = torax_config.geometry.build_provider.torax_mesh
    cell_centers = torax_mesh.cell_centers
    # Compare the initial core_profiles with the ids profiles
    init_core_profiles = sim_state.core_profiles
    np.testing.assert_allclose(
        init_core_profiles.T_e.value * 1e3,
        np.interp(
            cell_centers,
            rhon_in,
            core_profiles_in.profiles_1d[0].electrons.temperature,
        ),
        rtol=rtol,
        atol=atol,
        err_msg="Te profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.T_i.value * 1e3,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].t_i_average
        ),
        rtol=rtol,
        atol=atol,
        err_msg="Ti profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.n_e.value,
        np.interp(
            cell_centers,
            rhon_in,
            core_profiles_in.profiles_1d[0].electrons.density,
        ),
        rtol=rtol,
        atol=atol,
        err_msg="ne profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.psi.value,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].grid.psi
        ),
        rtol=rtol,
        atol=atol,
        err_msg="psi profile failed",
    )

  def test_imas_plasma_composition(self):
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")

    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    core_profiles_in = load_imas_data(path, "core_profiles", dir)
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
    # Modifying the input config profiles_conditions class
    core_profiles_data = core_profiles_from_IMAS(
        core_profiles_in,
    )
    config = update_dict(config, core_profiles_data)
    config["plasma_composition"]["impurity"]["impurity_mode"] = "n_e_ratios"
    torax_config = model_config.ToraxConfig.from_dict(config)

    assert torax_config.plasma_composition.impurity.species["Ne"] is not None
    np.testing.assert_allclose(
        torax_config.plasma_composition.impurity.species["Ne"].get_value(0.0),
        0.01,
    )
    np.testing.assert_allclose(
        torax_config.plasma_composition.main_ion["D"].get_value(0.0), 0.5
    )
    np.testing.assert_equal(
        torax_config.plasma_composition.get_main_ion_names(), ("D", "T")
    )
    np.testing.assert_equal(
        torax_config.plasma_composition.get_impurity_names(), ("Ne",)
    )


if __name__ == "__main__":
  absltest.main()
