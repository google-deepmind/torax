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

import copy
import os
from typing import Any, Final, Optional, Sequence
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from torax import simulation_app
from torax._src.config import config_loader
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.torax_pydantic import model_config
from torax.tests.test_lib import paths


_FAILED_TEST_OUTPUT_DIR: Final[str] = '/tmp/torax_failed_sim_test_outputs/'

# Default tolerances for checking sim results against references.
# np.allclose true if absolute(a - b) <= (atol + rtol * absolute(b))
# Therefore _ATOL = 0 restricts to a pure relative error check.
_RTOL = 1e-9
_ATOL = 0


class SimTestCase(parameterized.TestCase):
  """Base class for TestCases running TORAX sim.

  Contains useful functions for loading configs and checking sim results against
  references.
  """

  def setUp(self):
    super().setUp()

    self.test_data_dir = paths.test_data_dir()

  def _expected_results_path(self, test_name: str) -> str:
    return os.path.join(self.test_data_dir, f'{test_name}')

  def _get_config_dict(self, config_name: str) -> dict[str, Any]:
    """Returns a deepcopy of the config dict given the name of a module."""
    cfg = config_loader.import_module(
        os.path.join(self.test_data_dir, config_name)
    )
    return copy.deepcopy(cfg['CONFIG'])

  def _get_torax_config(self, config_name: str) -> model_config.ToraxConfig:
    """Returns a ToraxConfig given the name of a py file to build it."""
    return config_loader.build_torax_config_from_file(
        os.path.join(self.test_data_dir, config_name)
    )

  def _get_refs(
      self,
      ref_name: str,
      profiles: Sequence[str],
  ):
    """Gets reference values for the requested state profiles."""
    expected_results_path = self._expected_results_path(ref_name)
    self.assertTrue(os.path.exists(expected_results_path))
    data_tree = output.safe_load_dataset(expected_results_path)
    profiles_dataset = data_tree.children[output.PROFILES].dataset
    self.assertNotEmpty(profiles)
    ref_profiles = {
        profile: profiles_dataset[profile].to_numpy()
        for profile in profiles
    }
    ref_time = profiles_dataset[output.TIME].to_numpy()
    self.assertEqual(ref_time.shape[0], ref_profiles[profiles[0]].shape[0])
    return ref_profiles, ref_time

  def _check_profiles_vs_expected(
      self,
      t,
      ref_time,
      ref_profiles,
      rtol,
      atol,
      output_file=None,
      ds=None,
      write_output=True,
  ):
    """Raises an error if the input states and time do not match the refs."""
    chex.assert_rank(t, 1)
    history_length = t.shape[0]

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
        actual_value = (
            ds.children[output.PROFILES]
            .dataset[profile_name]
            .to_numpy()[step, :]
        )
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
            'Pos\tActual\tExpected\tRel Err\tMatch',
        ]
        for i in range(actual.shape[0]):
          match = np.allclose(actual[i], ref[i], rtol=rtol, atol=atol)
          rele = np.abs((actual[i] - ref[i])/ref[i])
          msg.append(f'{names[i]}\t{actual[i]}\t{ref[i]}\t{rele:e}\t{match}')
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
      # Write all outputs to tmp dirs, used for automated comparisons and
      # updates of references.
      if output_file is not None and ds is not None and write_output:
        _ = simulation_app.write_output_to_file(output_file, ds)

      raise AssertionError(final_msg)

  def _test_run_simulation(
      self,
      config_name: str,
      profiles: Sequence[str],
      ref_name: Optional[str] = None,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
      write_output: bool = True,
  ):
    """Integration test comparing to TORAX reference output."""
    if rtol is None:
      rtol = _RTOL
    if atol is None:
      atol = _ATOL

    torax_config = self._get_torax_config(config_name)
    output_xr, _ = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    output_file = _FAILED_TEST_OUTPUT_DIR + config_name[:-3] + '.nc'

    if ref_name is None:
      ref_name = f'{config_name[:-3]}.nc'

    ref_profiles, ref_time = self._get_refs(ref_name, profiles)

    self._check_profiles_vs_expected(
        t=output_xr.time.values,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=rtol,
        atol=atol,
        output_file=output_file,
        ds=output_xr,
        write_output=write_output,
    )
