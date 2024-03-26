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

"""Unit tests for torax.state and torax.initial_states."""

import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import config as config_lib
from torax import geometry
from torax import initial_states
from torax import state
from torax.tests.test_lib import torax_refs


class StateTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.state` module."""

  def setUp(self):
    super().setUp()

    # Make a State object in history mode, output by scan
    self.history_length = 2

    def make_hist(config, geo):
      initial_counter = jnp.array(0)

      def scan_f(counter: jax.Array, _) -> tuple[jax.Array, state.CoreProfiles]:
        core_profiles = initial_states.initial_core_profiles(config, geo)
        # Make one variable in the history track the value of the counter
        value = jnp.ones_like(core_profiles.temp_ion.value) * counter
        core_profiles = config_lib.recursive_replace(
            core_profiles, temp_ion={'value': value}
        )
        return counter + 1, core_profiles.history_elem()

      _, history = jax.lax.scan(
          scan_f,
          initial_counter,
          xs=None,
          length=self.history_length,
      )
      return history

    def make_history(config, geo):
      # Bind non-JAX arguments so it can be jitted
      bound = functools.partial(make_hist, config, geo)
      return jax.jit(bound)()

    self._make_history = make_history

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_sanity_check(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Make sure State.sanity_check can be called."""
    references = references_getter()
    basic_core_profiles = initial_states.initial_core_profiles(
        references.config,
        references.geo,
    )
    basic_core_profiles.sanity_check()

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_index(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Test State.index."""
    references = references_getter()
    history = self._make_history(references.config, references.geo)

    for i in range(self.history_length):
      self.assertEqual(i, history.index(i).temp_ion.value[0])

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_project(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Test State.project."""
    references = references_getter()
    history = self._make_history(references.config, references.geo)

    seed = 20230421
    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used
    weights = jax.random.normal(rng_state, (self.history_length,))
    del rng_state  # Make sure rng_state isn't accidentally re-used

    expected = jnp.dot(weights, jnp.arange(self.history_length))

    projected = history.project(weights)

    actual = projected.temp_ion.value[0]

    np.testing.assert_allclose(expected, actual)


class InitialStatesTest(parameterized.TestCase):
  """Unit tests for the `torax.initial_states` module."""

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    config = config_lib.Config(
        Ti_bound_right=27.7,
        Te_bound_right={0.0: 42.0, 1.0: 0.0},
        ne_bound_right=config_lib.InterpolationParam(
            {0.0: 0.1, 1.0: 2.0},
            interpolation_mode=config_lib.InterpolationMode.STEP,
        ),
    )
    core_profiles = initial_states.initial_core_profiles(
        config, geometry.build_circular_geometry(config)
    )
    np.testing.assert_allclose(
        core_profiles.temp_ion.right_face_constraint, 27.7
    )
    np.testing.assert_allclose(
        core_profiles.temp_el.right_face_constraint, 42.0
    )
    np.testing.assert_allclose(core_profiles.ne.right_face_constraint, 0.1)


if __name__ == '__main__':
  absltest.main()
