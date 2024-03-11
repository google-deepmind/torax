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

"""TestCase base class for running tests with sim.py."""

import functools
import importlib
import os
from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import chex
import h5py
import jax.numpy as jnp
import numpy as np
import torax
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.sources import source_profiles
from torax.stepper import nonlinear_theta_method
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import array_time_step_calculator
from torax.transport_model import transport_model as transport_model_lib


_TORAX_TO_PINT = {
    'temp_ion': 'Ti',
    'temp_el': 'Te',
    'psi': 'psi',
    's_face': 's',
    'q_face': 'q',
    'ne': 'ne',
}


_PYTHON_MODULE_PREFIX = '.tests.test_data.'
_PYTHON_CONFIG_PACKAGE = 'torax'


class SimTestCase(parameterized.TestCase):
  """Base class for TestCases running TORAX sim.

  Contains useful functions for loading configs and checking sim results against
  references.
  """

  rtol = 2e-3
  atol = 1e-11

  def setUp(self):
    super().setUp()

    src_dir = absltest.TEST_SRCDIR.value
    torax_dir = 'torax/'
    self.test_data_dir = os.path.join(src_dir, torax_dir, 'tests/test_data')

  def _expected_results_path(self, test_name: str) -> str:
    return os.path.join(self.test_data_dir, f'{test_name}.h5')

  def _get_config_module(
      self,
      config_name: str,
  ):
    """Returns an input Config from the name given."""
    test_config_path = os.path.join(self.test_data_dir, config_name)
    assert os.path.exists(test_config_path), test_config_path

    # Load config structure with test-case-specific values.
    assert config_name.endswith('.py'), config_name
    config_name_no_py = config_name[:-3]
    python_config_module = _PYTHON_MODULE_PREFIX + config_name_no_py
    return importlib.import_module(python_config_module, _PYTHON_CONFIG_PACKAGE)

  def _get_config(
      self,
      config_name: str,
  ) -> config_lib.Config:
    """Returns an input Config from the name given."""
    config_module = self._get_config_module(config_name)
    return config_module.get_config()

  def _get_geometry(
      self,
      config_name: str,
  ) -> geometry.Geometry:
    """Returns an input Config from the name given."""
    config_module = self._get_config_module(config_name)
    config = config_module.get_config()
    return config_module.get_geometry(config)

  def _get_sim(self, config_name: str) -> sim_lib.Sim:
    """Returns a Sim given the name of a py file to build it."""
    config_module = self._get_config_module(config_name)
    return config_module.get_sim()

  def _get_refs(
      self,
      ref_name: str,
      profiles: Sequence[str],
  ):
    """Gets reference values for the requested state profiles."""
    expected_results_path = self._expected_results_path(ref_name)
    self.assertTrue(os.path.exists(expected_results_path))

    with open(expected_results_path, mode='rb') as f:
      with h5py.File(f, 'r') as hf:
        self.assertNotEmpty(profiles)
        if 'Ti' in hf.keys():  # Determine if h5 file is PINT output
          ref_profiles = {
              profile: hf[_TORAX_TO_PINT[profile]][:] for profile in profiles
          }
        else:
          ref_profiles = {profile: hf[profile][:] for profile in profiles}
        ref_time = jnp.array(hf['t'])
        self.assertEqual(ref_time.shape[0], ref_profiles[profiles[0]].shape[0])
        return ref_profiles, ref_time

  def _check_profiles_vs_expected(
      self,
      state_history,
      t,
      ref_time,
      ref_profiles,
      rtol,
      atol,
  ):
    """Raises an error if the input states and time do not match the refs."""
    chex.assert_rank(t, 1)
    history_length = t.shape[0]
    self.assertEqual(state_history.temp_el.value.shape[0], t.shape[0])

    msgs = []
    mismatch_found = False
    err_norms = []

    for step, ref_t in enumerate(ref_time):
      if step >= t.shape[0]:
        break
      # Concatenate all information for the step, so we give a report on all
      # mistakes in a step simultaneously
      actual = [t[step : step + 1]]
      ref = [jnp.expand_dims(ref_t, axis=0)]
      names = ['t']
      for profile_name, ref_profile in ref_profiles.items():
        torax_var_history = state_history[profile_name]
        if isinstance(torax_var_history, torax.CellVariable):
          actual_value_history = torax_var_history.value
        else:
          actual_value_history = torax_var_history
        self.assertEqual(actual_value_history.shape[0], history_length)
        actual_value = actual_value_history[step, :]
        ref_value = ref_profile[step, :]
        with self.subTest(step=step, ref_profile=ref_profile):
          self.assertEqual(actual_value.shape, ref_value.shape)
        actual.append(actual_value)
        ref.append(ref_value)
        names.extend([f'{profile_name}_{i}' for i in range(ref_value.shape[0])])
      actual = jnp.concatenate(actual)
      ref = jnp.concatenate(ref)
      err_norms.append(np.sqrt(np.square(actual - ref).sum()))

      # Log mismatch if we find the first mismatch on this step
      mismatch_this_step = (not mismatch_found) and not np.allclose(
          actual,
          ref,
          rtol=rtol,
          atol=atol,
      )
      mismatch_found = mismatch_this_step or mismatch_found
      # Log mismatch if there is a known mismatch and this is the last step
      # (so we get an idea of the error at steady state)
      log_last_step = mismatch_found and step == len(ref_time) - 1

      should_log_mismatch = mismatch_this_step or log_last_step

      if should_log_mismatch:
        # These messages are long so we print just the first mismatch and the
        # mismatch at the last step
        msg = [
            f'Mismatch at step {step}, t = {ref_time[step]}',
            'Pos\tActual\tExpected\tAbs Err\tMatch',
        ]
        for i in range(actual.shape[0]):
          match = np.allclose(actual[i], ref[i], rtol=rtol)
          abse = np.abs(actual[i] - ref[i])
          msg.append(f'{names[i]}\t{actual[i]}\t{ref[i]}\t{abse:e}\t{match}')
        msg = '\n'.join(msg)
        msgs.append(msg)

    if history_length < ref_time.shape[0]:
      msg = (
          f'Ended early, with final time %{t[-1]}. '
          f'Remaining reference time: %{ref_time[t.shape[0]:]}.'
      )
      msgs.append(msg)

    if t.shape[0] > ref_time.shape[0]:
      excess = t.shape[0] - ref_time.shape[0]
      msg = (
          'Used too many steps. '
          f'Extra {excess} steps are {t[ref_time.shape[0]:]}.'
      )
      msgs.append(msg)

    if msgs:
      # Insert a message about error increasing over time before other messages
      msg = ['Error over time: \tIdx\tError norm']
      msg.extend([f'\t{i}\t{e}' for i, e in enumerate(err_norms)])
      msg = '\n'.join(msg)

      msgs.insert(0, msg)

      final_msg = '\n'.join(msgs)

      raise AssertionError(final_msg)

  def _test_torax_sim(
      self,
      config_name: str,
      ref_name: str,
      profiles: Sequence[str],
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
      use_ref_time: bool = False,
  ):
    """Integration test comparing to TORAX reference output.

    Args:
      config_name: Name of py config to load. (Leave off dir path, include
        ".py")
      ref_name: Name of h5 reference solution to load. (Leave off dir path,
        ".h5")
      profiles: List of names of variables to check.
      rtol: Optional float, to override the class level rtol.
      atol: Optional float, to override the class level atol.
      use_ref_time: If True, locks to time steps calculated by reference.
    """

    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol

    sim = self._get_sim(config_name)

    # Load reference profiles
    ref_profiles, ref_time = self._get_refs(ref_name, profiles)

    if use_ref_time:
      time_step_calculator = array_time_step_calculator.ArrayTimeStepCalculator(
          ref_time
      )
      sim = sim_lib.Sim(
          time_step_calculator=time_step_calculator,
          initial_state=sim.initial_state,
          geometry_provider=sim.geometry_provider,
          dynamic_config_slice_provider=sim.dynamic_config_slice_provider,
          static_config_slice=sim.static_config_slice,
          stepper=sim.stepper,
          transport_model=sim.transport_model,
          step_fn=sim.step_fn,
      )

    torax_outputs = sim.run()
    state_history, _ = state_lib.build_history_from_outputs(torax_outputs)
    t = state_lib.build_time_history_from_outputs(torax_outputs)

    self._check_profiles_vs_expected(
        state_history=state_history,
        t=t,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=rtol,
        atol=atol,
    )


def make_frozen_optimizer_stepper(
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    config: config_lib.Config,
) -> stepper_lib.Stepper:
  """Makes an optimizer stepper with frozen coefficients.

  Under these conditions, we can test that the optimizer behaves the same as
  the linear solver.

  Args:
    transport_model: Transport model.
    sources: TORAX sources/sinks used to compute profile terms in the state
      evolution equations.
    config: General TORAX config.

  Returns:
    Stepper: the stepper.
  """
  # Get the dynamic config for the start of the simulation.
  dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
  callback_builder = functools.partial(
      sim_lib.FrozenCoeffsCallback,
      dynamic_config_slice=dynamic_config_slice,
  )
  return nonlinear_theta_method.OptimizerThetaMethod(
      transport_model,
      sources=sources,
      callback_class=callback_builder,  # pytype: disable=wrong-arg-types
  )
