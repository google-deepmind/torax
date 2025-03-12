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

"""TORAX integration tests.

These are full integration tests that run the simulation and compare to a
previously executed TORAX reference:
"""

import os
from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import output
from torax import sim as sim_lib
from torax import state
from torax.config import build_sim as build_sim_lib
from torax.config import numerics as numerics_lib
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.tests.test_lib import sim_test_case
from torax.time_step_calculator import chi_time_step_calculator
from torax.transport_model import constant as constant_transport_model
import xarray as xr

_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


# Sim tests. Note that if any test data file does not have the same name as the
# test name, then the nonstandard mapping must be added to _REF_MAP_OVERRIDES in
# tests/test_lib/__init__.py
class SimTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim."""

  @parameterized.named_parameters(
      # Tests explicit solver
      (
          'test_explicit',
          'test_explicit.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests implicit solver with theta=0.5 (Crank-Nicolson)
      # Low tolerance since solver parameters are different
      (
          'test_crank_nicolson',
          'test_crank_nicolson.py',
          ('temp_ion', 'temp_el'),
          2e-1,
      ),
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test_implicit',
          'test_implicit.py',
          _ALL_PROFILES,
          0,
      ),
      # Test ion-electron heat exchange at low density
      (
          'test_qei',
          'test_qei.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests pedestal internal boundary condition
      (
          'test_pedestal',
          'test_pedestal.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CGM model heat transport only
      (
          'test_cgmheat',
          'test_cgmheat.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests BgB model heat transport only
      (
          'test_bohmgyrobohm_all',
          'test_bohmgyrobohm_all.py',
          _ALL_PROFILES,
          0,
      ),
      # Test that we are able to reproduce FiPy's behavior in a case where
      # FiPy is unstable
      (
          'test_semiimplicit_convection',
          'test_semiimplicit_convection.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests QLKNN model, heat transport only
      (
          'test_qlknnheat',
          'test_qlknnheat.py',
          _ALL_PROFILES,
          0,
          1e-11,
      ),
      # Tests fixed_dt timestep
      (
          'test_fixed_dt',
          'test_fixed_dt.py',
          _ALL_PROFILES,
          0,
          1e-11,
      ),
      # Tests current diffusion
      (
          'test_psiequation',
          'test_psiequation.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined current diffusion + heat transport with QLKNN
      (
          'test_psi_and_heat',
          'test_psi_and_heat.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests implementation of use_absolute_current
      (
          'test_absolute_generic_current_source',
          'test_absolute_generic_current_source.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests that Newton-Raphson gets the same result as the linear solver
      # when using linear initial guess and 0 iterations
      # Making sure to use a test involving Pereverzev-Corrigan for this,
      # since we do want it in the linear initial guess.
      (
          'test_newton_raphson_zeroiter',
          'test_newton_raphson_zeroiter.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current with heat+current-diffusion. QLKNN model
      (
          'test_bootstrap',
          'test_bootstrap.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests heat+current-diffusion+particle transport with constant transport
      (
          'test_psi_heat_dens',
          'test_psi_heat_dens.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources with constant transport. No particle source
      (
          'test_particle_sources_constant',
          'test_particle_sources_constant.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests all particle sources including particle source, with CGM transport
      (
          'test_particle_sources_cgm',
          'test_particle_sources_cgm.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests specifying a prescribed time-varying arbitrary current profile
      (
          'test_prescribed_generic_current_source',
          'test_prescribed_generic_current_source.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. CGM transport, heat+particle+psi transport
      (
          'test_fusion_power',
          'test_fusion_power.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. QLKNN transport, heat+particle+psi transport.
      (
          'test_all_transport_fusion_qlknn',
          'test_all_transport_fusion_qlknn.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry. Implicit solver. Heat transport only.
      (
          'test_chease',
          'test_chease.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests EQDSK geometry. QLKNN, predictor-corrector, all transport.
      (
          'test_eqdsk',
          'test_eqdsk.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Ohmic electron heat source. CHEASE geometry.
      (
          'test_ohmic_power',
          'test_ohmic_power.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Bremsstrahlung heat sink. CHEASE geometry.
      (
          'test_bremsstrahlung',
          'test_bremsstrahlung.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Bremsstrahlung heat sink with time dependent Zimp and Zeff. CHEASE
      (
          'test_bremsstrahlung_time_dependent_Zimp',
          'test_bremsstrahlung_time_dependent_Zimp.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests ion-electron heat exchange test at high density. CHEASE geometry.
      (
          'test_qei_chease_highdens',
          'test_qei_chease_highdens.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from parameters. CHEASE geometry.
      (
          'test_psichease_ip_parameters',
          'test_psichease_ip_parameters.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from CHEASE.
      (
          'test_psichease_ip_chease',
          'test_psichease_ip_chease.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry with psi initialized from prescribed jtot.
      (
          'test_psichease_prescribed_jtot',
          'test_psichease_prescribed_jtot.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry with psi initialized from prescribed johm.
      (
          'test_psichease_prescribed_johm',
          'test_psichease_prescribed_johm.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests time-dependent pedestal, Ptot, Ip. CHEASE geometry. QLKNN.
      (
          'test_timedependence',
          'test_timedependence.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests prescribed time-dependent ne (tied to GW frac with evolving Ip).
      (
          'test_prescribed_timedependent_ne',
          'test_prescribed_timedependent_ne.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle transport with QLKNN. De scaled from chie.
      # CHEASE geometry
      (
          'test_ne_qlknn_defromchie',
          'test_ne_qlknn_defromchie.py',
          _ALL_PROFILES,
          1e-8,
      ),
      # Tests particle transport with QLKNN. Deff+Veff model. CHEASE geometry.
      (
          'test_ne_qlknn_deff_veff',
          'test_ne_qlknn_deff_veff.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Crank-Nicholson with particle transport and QLKNN. Deff+Veff
      (
          'test_all_transport_crank_nicolson',
          'test_all_transport_crank_nicolson.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Pereverzev-Corrigan method for density. CHEASE geometry. QLKNN.
      # De scaled from chie.
      (
          'test_pc_method_ne',
          'test_pc_method_ne.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-baseline-like config. Linear solver.
      (
          'test_iterbaseline_mockup',
          'test_iterbaseline_mockup.py',
          _ALL_PROFILES,
          1e-10,
      ),
      # Tests full integration for ITER-hybrid-like config. Linear solver.
      (
          'test_iterhybrid_mockup',
          'test_iterhybrid_mockup.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-hybrid-like config.
      # Predictor-corrector solver.
      (
          'test_iterhybrid_predictor_corrector',
          'test_iterhybrid_predictor_corrector.py',
          _ALL_PROFILES,
          0,
      ),
      # ITERhybrid_predictor_corrector with EQDSK geometry.
      # See https://github.com/google-deepmind/torax/pull/482 for a plot
      # of the CHEASE vs EQDSK sim test comparison.
      (
          'test_iterhybrid_predictor_corrector_eqdsk',
          'test_iterhybrid_predictor_corrector_eqdsk.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with clipped QLKNN inputs.
      (
          'test_iterhybrid_predictor_corrector_clip_inputs',
          'test_iterhybrid_predictor_corrector_clip_inputs.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with non-constant Zeff profile.
      (
          'test_iterhybrid_predictor_corrector_zeffprofile',
          'test_iterhybrid_predictor_corrector_zeffprofile.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with a time-dependent isotope mix.
      (
          'test_iterhybrid_predictor_corrector_timedependent_isotopes',
          'test_iterhybrid_predictor_corrector_timedependent_isotopes.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with tungsten.
      (
          'test_iterhybrid_predictor_corrector_tungsten',
          'test_iterhybrid_predictor_corrector_tungsten.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with ECCD Lin Liu model.
      (
          'test_iterhybrid_predictor_corrector_ec_linliu',
          'test_iterhybrid_predictor_corrector_ec_linliu.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with constant fraction of Pin radiation
      (
          'test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation',
          'test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with constant pressure pedestal model.
      (
          'test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped',
          'test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with cyclotron radiation heat sink
      (
          'test_iterhybrid_predictor_corrector_cyclotron',
          'test_iterhybrid_predictor_corrector_cyclotron.py',
          _ALL_PROFILES,
          0,
          1e-8,
      ),
      # Tests Newton-Raphson nonlinear solver for ITER-hybrid-like-config
      (
          'test_iterhybrid_newton',
          'test_iterhybrid_newton.py',
          _ALL_PROFILES,
          5e-7,
      ),
      # Tests current and density rampup for for ITER-hybrid-like-config
      # using Newton-Raphson. Only case which reverts to coarse_tol for several
      # timesteps (with negligible impact on results compared to full tol).
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
          _ALL_PROFILES,
          0,
          1e-6,
      ),
      # Tests time-dependent circular geometry.
      (
          'test_time_dependent_circular_geo',
          'test_time_dependent_circular_geo.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_before',
          'test_changing_config_before.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_after',
          'test_changing_config_after.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip from parameters and psi from CHEASE.
      (
          'test_psichease_ip_parameters_vloop',
          'test_psichease_ip_parameters_vloop.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip and psi from CHEASE.
      (
          'test_psichease_ip_chease_vloop',
          'test_psichease_ip_chease_vloop.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip from parameters and psi from nu formula.
      (
          'test_psichease_prescribed_jtot_vloop',
          'test_psichease_prescribed_jtot_vloop.py',
          _ALL_PROFILES,
          0,
      ),
  )
  def test_torax_sim(
      self,
      config_name: str,
      profiles: Sequence[str],
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Integration test comparing to reference output from TORAX."""
    # The @parameterized decorator removes the `test_torax_sim` method,
    # so we separate the actual functionality into a helper method that will
    # not be removed.
    self._test_torax_sim(
        config_name,
        profiles,
        rtol=rtol,
        atol=atol,
    )

  def test_fail(self):
    """Test that the integration tests can actually fail."""

    # Run test_qei but pass in the reference result from test_implicit.
    with self.assertRaises(AssertionError):
      self._test_torax_sim(
          'test_qei.py',
          ('temp_ion', 'temp_el'),
          ref_name='test_implicit.nc',
          write_output=False,
      )

  def test_no_op(self):
    """Tests that running the stepper with all equations off is a no-op."""

    runtime_params = runtime_params_lib.GeneralRuntimeParams(
        numerics=numerics_lib.Numerics(
            t_final=0.1,
            ion_heat_eq=False,
            el_heat_eq=False,
            current_eq=False,
        ),
    )

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    geo_provider = geometry_provider.ConstantGeometryProvider(
        geometry_pydantic_model.CircularConfig().build_geometry()
    )

    sim = sim_lib.Sim.create(
        runtime_params=runtime_params,
        geometry_provider=geo_provider,
        stepper=stepper_pydantic_model.Stepper(),
        transport_model_builder=constant_transport_model.ConstantTransportModelBuilder(),
        sources=source_pydantic_model.Sources(),
        time_step_calculator=time_step_calculator,
        pedestal=pedestal_pydantic_model.Pedestal(),
    )

    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs, sim.source_models)

    history_length = history.core_profiles.temp_ion.value.shape[0]
    self.assertEqual(history_length, history.times.shape[0])
    self.assertGreater(history.times[-1], runtime_params.numerics.t_final)

    for torax_profile in _ALL_PROFILES:
      profile_history = history.core_profiles[torax_profile]
      # This is needed for CellVariable but not face variables
      if hasattr(profile_history, 'value'):
        profile_history = profile_history.value
      first_profile = profile_history[0]
      if not all(
          [np.all(profile == first_profile) for profile in profile_history]
      ):
        for i in range(1, len(profile_history)):
          # Most profiles should be == but jtot, q_face, and s_face can be
          # merely allclose because they are recalculated on each step.
          if not np.allclose(profile_history[i], first_profile):
            msg = (
                'Profile changed over time despite all equations being '
                'disabled.\n'
                f'Profile name: {torax_profile}\n'
                f'Initial value: {first_profile}\n'
                f'Failing time index: {i}\n'
                f'Failing value: {profile_history[i]}\n'
                f'Equality mask: {profile_history[i] == first_profile}\n'
                f'Diff: {profile_history[i] - first_profile}\n'
            )
            raise AssertionError(msg)

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
        output.IP_PROFILE_FACE,
        output.NREF,
        output.Q_FACE,
        output.S_FACE,
        output.J_BOOTSTRAP,
        output.J_BOOTSTRAP_FACE,
        output.JOHM,
        output.EXTERNAL_CURRENT,
        output.JTOT,
        output.JTOT_FACE,
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

    # Load in the reference core profiles.
    Ip_total = ref_profiles[output.IP_PROFILE_FACE][index, -1] / 1e6
    temp_el = ref_profiles[output.TEMP_EL][index, :]
    temp_el_bc = ref_profiles[output.TEMP_EL_RIGHT_BC][index]
    temp_ion = ref_profiles[output.TEMP_ION][index, :]
    temp_ion_bc = ref_profiles[output.TEMP_ION_RIGHT_BC][index]
    ne = ref_profiles[output.NE][index, :]
    ne_bound_right = ref_profiles[output.NE_RIGHT_BC][index]
    psi = ref_profiles[output.PSI][index, :]

    # Override the dynamic runtime params with the loaded values.
    dynamic_runtime_params_slice.profile_conditions.Ip_tot = Ip_total
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

    # Get initial core profiles for the overriden dynamic runtime params.
    initial_state = sim_lib.get_initial_state(
        sim.static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        sim.step_fn,
    )

    # Check for agreement with the reference core profiles.
    verify_core_profiles(ref_profiles, index, initial_state.core_profiles)

    if halfway:
      # Run sim till the end and check that final core profiles match reference.
      initial_state.t = ref_time[index]
      sim._initial_state = initial_state  # pylint: disable=protected-access
      sim_outputs = sim.run()
      final_core_profiles = sim_outputs.sim_history[-1].core_profiles
      verify_core_profiles(ref_profiles, -1, final_core_profiles)
    # pylint: enable=invalid-name

  def test_nans_trigger_error(self):
    """Verify that NaNs in profile evolution triggers early stopping and an error."""

    config_module = self._get_config_module('test_iterhybrid_makenans.py')
    sim = build_sim_lib.build_sim_from_config(config_module.CONFIG)
    sim_outputs = sim.run()

    state_history = output.StateHistory(sim_outputs, sim.source_models)
    self.assertEqual(state_history.sim_error, state.SimError.NAN_DETECTED)
    assert (
        state_history.times[-1]
        < config_module.CONFIG['runtime_params']['numerics']['t_final']
    )

  def test_restart_sim_from_file(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    restart_config = 'test_iterhybrid_rampup_restart.py'
    sim = self._get_sim(restart_config)
    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs, sim.source_models)
    data_tree_restart = history.simulation_output_to_xr()

    # Load the reference dataset.
    datatree_ref = output.load_state_file(
        os.path.join(self.test_data_dir, test_config_state_file)
    )

    # Stitch the restart state file to the beginning of the reference dataset.
    datatree_new = output.stitch_state_files(
        sim.file_restart, data_tree_restart
    )

    # Check equality for all time-dependent variables.
    def check_equality(ds1: xr.Dataset, ds2: xr.Dataset):
      for var_name in ds1.data_vars:
        if 'time' in ds1[var_name].dims:
          with self.subTest(var_name=var_name):
            np.testing.assert_allclose(
                ds1[var_name].values,
                ds2[var_name].values,
                err_msg=f'Mismatch for {var_name} in restart test',
                rtol=1e-6,
            )

    xr.map_over_datasets(check_equality, datatree_ref, datatree_new)

  def test_update(self):
    sim = self._get_sim('test_iterhybrid_predictor_corrector.py')
    new_config = self._get_config_module(
        'test_iterhybrid_predictor_corrector_eqdsk.py'
    ).CONFIG
    sim.update_base_components(
        geometry_provider=geometry_pydantic_model.Geometry.from_dict(
            new_config['geometry']
        ).build_provider
    )
    sim_outputs = sim.run()

    # Extract core profiles history for analysis against references
    history = output.StateHistory(sim_outputs, sim.source_models)
    ref_profiles, ref_time = self._get_refs(
        'test_iterhybrid_predictor_corrector_eqdsk.nc', _ALL_PROFILES
    )

    self._check_profiles_vs_expected(
        core_profiles=history.core_profiles,
        t=history.times,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=self.rtol,
        atol=self.atol,
    )

  def test_update_new_mesh(self):
    sim = self._get_sim('test_iterhybrid_rampup.py')
    with self.assertRaisesRegex(ValueError, 'different mesh'):
      sim.update_base_components(
          geometry_provider=geometry_pydantic_model.Geometry.from_dict(
              {'geometry_type': 'circular', 'n_rho': 10}
          ).build_provider
      )


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
      core_profiles.currents.jtot, ref_profiles[output.JTOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot_face, ref_profiles[output.JTOT_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap_face,
      ref_profiles[output.J_BOOTSTRAP_FACE][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.external_current_source,
      ref_profiles[output.EXTERNAL_CURRENT][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm, ref_profiles[output.JOHM][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.I_bootstrap,
      ref_profiles[output.I_BOOTSTRAP][index],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip_profile_face,
      ref_profiles[output.IP_PROFILE_FACE][index, :],
  )


if __name__ == '__main__':
  absltest.main()
