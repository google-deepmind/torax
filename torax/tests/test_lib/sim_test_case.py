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

from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
import torax
from torax import sim as sim_lib
from torax import simulation_app
from torax import state as state_lib
from torax.config import build_sim
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.stepper import nonlinear_theta_method
from torax.stepper import stepper as stepper_lib
from torax.tests import test_lib
from torax.tests.test_lib import paths
from torax.time_step_calculator import array_time_step_calculator
from torax.transport_model import runtime_params as transport_params_lib
from torax.transport_model import transport_model as transport_model_lib
import xarray as xr

_PYTHON_MODULE_PREFIX = '.tests.test_data.'
_PYTHON_CONFIG_PACKAGE = 'torax'
_FAILED_TEST_OUTPUT_DIR = '/tmp/torax_failed_sim_test_outputs/'


class SimTestCase(parameterized.TestCase):
  """Base class for TestCases running TORAX sim.

  Contains useful functions for loading configs and checking sim results against
  references.
  """

  rtol = 2e-3
  atol = 1e-10

  def setUp(self):
    super().setUp()

    self.test_data_dir = paths.test_data_dir()

  def _expected_results_path(self, test_name: str) -> str:
    return os.path.join(self.test_data_dir, f'{test_name}')

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

  def _get_sim(self, config_name: str) -> sim_lib.Sim:
    """Returns a Sim given the name of a py file to build it."""
    config_module = self._get_config_module(config_name)
    if hasattr(config_module, 'get_sim'):
      # The config module likely uses the "advanced" configuration setup with
      # python functions defining all the Sim object attributes.
      return config_module.get_sim()
    elif hasattr(config_module, 'CONFIG'):
      # The config module is using the "basic" configuration setup with a single
      # CONFIG dictionary defining everything.
      # This CONFIG needs to be built into an actual Sim object.
      return build_sim.build_sim_from_config(config_module.CONFIG)
    else:
      raise ValueError(
          f'Config module {config_name} must either define a get_sim() method'
          ' or a CONFIG dictionary.'
      )

  def _get_refs(
      self,
      ref_name: str,
      profiles: Sequence[str],
  ):
    """Gets reference values for the requested state profiles."""
    expected_results_path = self._expected_results_path(ref_name)
    self.assertTrue(os.path.exists(expected_results_path))
    ds = xr.open_dataset(expected_results_path)
    self.assertNotEmpty(profiles)
    ref_profiles = {profile: ds[profile].to_numpy() for profile in profiles}
    if 'time' in ds:
      ref_time = ds['time'].to_numpy()
    else:
      ref_time = ds['t'].to_numpy()
    self.assertEqual(ref_time.shape[0], ref_profiles[profiles[0]].shape[0])
    return ref_profiles, ref_time

  def _check_profiles_vs_expected(
      self,
      core_profiles,
      t,
      ref_time,
      ref_profiles,
      rtol,
      atol,
      output_dir=None,
      ds=None,
  ):
    """Raises an error if the input states and time do not match the refs."""
    chex.assert_rank(t, 1)
    history_length = t.shape[0]
    self.assertEqual(core_profiles.temp_el.value.shape[0], t.shape[0])

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
        torax_var_history = core_profiles[profile_name]
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
          match = np.allclose(actual[i], ref[i], rtol=rtol, atol=atol)
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
      # Write all outputs to tmp dirs, used for automated comparisons and
      # updates of references.
      if output_dir is not None and ds is not None:
        _ = simulation_app.write_simulation_output_to_file(output_dir, ds)

      raise AssertionError(final_msg)

  def _test_torax_sim(
      self,
      config_name: str,
      profiles: Sequence[str],
      ref_name: Optional[str] = None,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
      use_ref_time: bool = False,
  ):
    """Integration test comparing to TORAX reference output.

    Args:
      config_name: Name of py config to load. (Leave off dir path, include
        ".py")
      profiles: List of names of variables to check.
      ref_name: Name of reference filename to load. (Leave off dir path)
      rtol: Optional float, to override the class level rtol.
      atol: Optional float, to override the class level atol.
      use_ref_time: If True, locks to time steps calculated by reference.
    """

    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol

    sim = self._get_sim(config_name)

    if ref_name is None:
      ref_name = test_lib.get_data_file(config_name[:-3])

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
          dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
          static_runtime_params_slice=sim.static_runtime_params_slice,
          stepper=sim.stepper,
          transport_model=sim.transport_model,
          step_fn=sim.step_fn,
          source_models_builder=sim.source_models_builder,
      )

    # Build geo needed for output generation
    geo = sim.geometry_provider(sim.initial_state.t)

    # Run full simulation
    torax_outputs = sim.run()

    # Extract core profiles history for analysis against references
    core_profiles, _, _ = state_lib.build_history_from_states(torax_outputs)
    t = state_lib.build_time_history_from_states(torax_outputs)

    ds = simulation_app.simulation_output_to_xr(torax_outputs, geo)
    output_dir = _FAILED_TEST_OUTPUT_DIR + config_name[:-3]

    self._check_profiles_vs_expected(
        core_profiles=core_profiles,
        t=t,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=rtol,
        atol=atol,
        output_dir=output_dir,
        ds=ds,
    )


def make_frozen_optimizer_stepper(
    transport_model: transport_model_lib.TransportModel,
    source_models_builder: source_models_lib.SourceModelsBuilder,
    source_models: source_models_lib.SourceModels,
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    transport_params: transport_params_lib.RuntimeParams,
) -> stepper_lib.Stepper:
  """Makes an optimizer stepper with frozen coefficients.

  Under these conditions, we can test that the optimizer behaves the same as
  the linear solver.

  Args:
    transport_model: Transport model.
    source_models_builder: Holds the runtime_params for source_models
    source_models: TORAX sources/sinks used to compute profile terms in the
      state evolution equations.
    runtime_params: General TORAX runtime input parameters.
    transport_params: Runtime params for the transport model.

  Returns:
    Stepper: the stepper.
  """
  # Get the dynamic runtime params for the start of the simulation.
  dynamic_runtime_params_slice = (
      runtime_params_slice.build_dynamic_runtime_params_slice(
          runtime_params=runtime_params,
          transport=transport_params,
          sources=source_models_builder.runtime_params,
      )
  )
  callback_builder = functools.partial(
      sim_lib.FrozenCoeffsCallback,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
  )
  return nonlinear_theta_method.OptimizerThetaMethod(
      transport_model,
      source_models=source_models,
      callback_class=callback_builder,  # pytype: disable=wrong-arg-types
  )


def make_frozen_newton_raphson_stepper(
    transport_model: transport_model_lib.TransportModel,
    source_models: source_models_lib.SourceModels,
    runtime_params: general_runtime_params.GeneralRuntimeParams,
) -> stepper_lib.Stepper:
  """Makes a Newton Raphson stepper with frozen coefficients.

  Under these conditions, we can test that the nonlinear stepper behaves the
  same as
  the linear solver.

  Args:
    transport_model: Transport model.
    source_models: TORAX sources/sinks used to compute profile terms in the
      state evolution equations.
    runtime_params: General TORAX runtime input parameters.

  Returns:
    Stepper: the stepper.
  """
  # Get the dynamic runtime params for the start of the simulation.
  dynamic_runtime_params_slice = (
      runtime_params_slice.build_dynamic_runtime_params_slice(runtime_params)
  )
  callback_builder = functools.partial(
      sim_lib.FrozenCoeffsCallback,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
  )
  functools.partial(sim_lib.FrozenCoeffsCallback, runtime_params=runtime_params)
  return nonlinear_theta_method.NewtonRaphsonThetaMethod(
      transport_model,
      source_models=source_models,
      callback_class=callback_builder,  # pytype: disable=wrong-arg-types
  )
