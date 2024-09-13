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

"""Tests that TORAX can be run with compilation disabled."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import output
from torax import sim as sim_lib
from torax.tests.test_lib import paths
from torax.tests.test_lib import sim_test_case
import xarray as xr


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimTest(sim_test_case.SimTestCase):
  """Restart integration tests for torax.sim."""

  # pylint: disable=invalid-name
  @parameterized.product(
      test_config=(
          'test_psi_heat_dens',
          'test_psichease_ip_parameters',
          'test_psichease_ip_chease',
          'test_psichease_prescribed_jtot',
          'test_psichease_prescribed_johm',
          'test_iterhybrid_rampup',
      ),
      halfway=(False, True),
  )
  def test_core_profiles_are_recomputable(self, test_config, halfway):
    """Tests that core profiles from a previous run are recomputable.

    In this test we:
    - Load up a reference file and build a sim from its config.
    - Get profile values from either halfway or final time of the sim.
    - Override the dynamic runtime params slice with values from the reference.
    - Check that the initial core_profiles are equal.
    - In the case of loading from halfway, run the sim to the end and also check
    against reference.

    Args:
      test_config: the config id under test.
      halfway: Whether to load from halfway (or the end if not) to test in case
        there is different behaviour for the final step.
    """
    profiles = [
        output.TEMP_ION,
        output.TEMP_ION_RIGHT_BC,
        output.TEMP_EL,
        output.TEMP_EL_RIGHT_BC,
        output.NE,
        output.NI,
        output.NE_RIGHT_BC,
        output.NI_RIGHT_BC,
        output.PSI,
        output.PSIDOT,
        output.IP,
        output.NREF,
        output.Q_FACE,
        output.S_FACE,
        output.J_BOOTSTRAP,
        output.JOHM,
        output.CORE_PROFILES_JEXT,
        output.JTOT,
        output.JTOT_FACE,
        output.JEXT_FACE,
        output.JOHM_FACE,
        output.J_BOOTSTRAP_FACE,
        output.I_BOOTSTRAP,
        output.SIGMA,
    ]
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    if halfway:
      index = len(ref_time) // 2
    else:
      index = -1
    loading_time = ref_time[index]

    # Build the sim and runtime params at t=`loading_time`.
    sim = self._get_sim(test_config + '.py')
    geo = sim.geometry_provider(t=loading_time)
    dynamic_runtime_params_slice = sim.dynamic_runtime_params_slice_provider(
        t=loading_time,
    )
    source_models = sim.source_models_builder()

    # Load in the reference core profiles.
    Ip = ref_profiles[output.IP][index]
    temp_el = ref_profiles[output.TEMP_EL][index, :]
    temp_el_bc = ref_profiles[output.TEMP_EL_RIGHT_BC][index]
    temp_ion = ref_profiles[output.TEMP_ION][index, :]
    temp_ion_bc = ref_profiles[output.TEMP_ION_RIGHT_BC][index]
    ne = ref_profiles[output.NE][index, :]
    ne_bound_right = ref_profiles[output.NE_RIGHT_BC][index]
    psi = ref_profiles[output.PSI][index, :]

    # Override the dynamic runtime params with the loaded values.
    dynamic_runtime_params_slice.profile_conditions.Ip = Ip
    dynamic_runtime_params_slice.profile_conditions.Te = temp_el
    dynamic_runtime_params_slice.profile_conditions.Te_bound_right = temp_el_bc
    dynamic_runtime_params_slice.profile_conditions.Ti = temp_ion
    dynamic_runtime_params_slice.profile_conditions.Ti_bound_right = temp_ion_bc
    dynamic_runtime_params_slice.profile_conditions.ne = ne
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right = (
        ne_bound_right
    )
    dynamic_runtime_params_slice.profile_conditions.psi = psi
    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not fGW).
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW = (
        False
    )
    dynamic_runtime_params_slice.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute = (
        True
    )
    # Additionally we want to avoid normalizing to nbar.
    dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar = False

    # Get initial core profiles for the overridden dynamic runtime params.
    initial_state = sim_lib.get_initial_state(
        dynamic_runtime_params_slice,
        geo,
        source_models,
        sim.time_step_calculator,
    )

    # Check for agreement with the reference core profiles.
    verify_core_profiles(ref_profiles, index, initial_state.core_profiles)

    if halfway:
      # Run sim till the end and check that final core profiles match reference.
      initial_state.t = ref_time[index]
      step_fn = sim_lib.SimulationStepFn(
          stepper=sim.stepper,
          time_step_calculator=sim.time_step_calculator,
          transport_model=sim.transport_model,
      )
      sim_outputs = sim_lib.run_simulation(
          static_runtime_params_slice=sim.static_runtime_params_slice,
          dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
          geometry_provider=sim.geometry_provider,
          initial_state=initial_state,
          time_step_calculator=sim.time_step_calculator,
          step_fn=step_fn,
      )
      final_core_profiles = sim_outputs.sim_history[-1].core_profiles
      verify_core_profiles(ref_profiles, -1, final_core_profiles)
    # pylint: enable=invalid-name

  def test_restart_sim_from_file(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    expected_ds = output.load_state_file(
        os.path.join(paths.test_data_dir(), test_config_state_file)
    )
    restart_config = 'test_iterhybrid_rampup_restart.py'
    sim = self._get_sim(restart_config)

    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs)
    ds = history.simulation_output_to_xr(
        sim.geometry_provider(t=sim.initial_state.t),
        sim.file_restart,
    )
    xr.testing.assert_allclose(ds, expected_ds)


def verify_core_profiles(ref_profiles, index, core_profiles):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.temp_el.value, ref_profiles[output.TEMP_EL][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.temp_ion.value, ref_profiles[output.TEMP_ION][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.value, ref_profiles[output.NE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.right_face_constraint,
      ref_profiles[output.NE_RIGHT_BC][index],
  )
  np.testing.assert_allclose(
      core_profiles.psi.value, ref_profiles[output.PSI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value, ref_profiles[output.PSIDOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.value, ref_profiles[output.NI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.right_face_constraint,
      ref_profiles[output.NI_RIGHT_BC][index],
  )

  np.testing.assert_allclose(
      core_profiles.q_face, ref_profiles[output.Q_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.s_face, ref_profiles[output.S_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.nref, ref_profiles[output.NREF][index]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap,
      ref_profiles[output.J_BOOTSTRAP][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap_face,
      ref_profiles[output.J_BOOTSTRAP_FACE][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot, ref_profiles[output.JTOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot_face, ref_profiles[output.JTOT_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.jext,
      ref_profiles[output.CORE_PROFILES_JEXT][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.jext_face, ref_profiles[output.JEXT_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm, ref_profiles[output.JOHM][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm_face, ref_profiles[output.JOHM_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.I_bootstrap,
      ref_profiles[output.I_BOOTSTRAP][index],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip, ref_profiles[output.IP][index]
  )


if __name__ == '__main__':
  absltest.main()
