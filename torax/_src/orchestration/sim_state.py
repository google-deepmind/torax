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

"""Full simulator state to be used for orchestration."""
from absl import logging
import chex
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.geometry import geometry
from torax._src.sources import source_profiles


@chex.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves core_profiles which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which may evolve on each simulation step, such
  as sources and transport.

  This class includes both core_profiles and these additional elements.

  Attributes:
    t: time coordinate.
    dt: timestep interval.
    core_profiles: Core plasma profiles at time t.
    core_transport: Core plasma transport coefficients computed at time t.
    core_sources: Profiles for all sources/sinks. These are the profiles that
      are used to calculate the coefficients for the t+dt time step. For the
      explicit sources, these are calculated at the start of the time step, so
      are the values at time t. For the implicit sources, these are the most
      recent guess for time t+dt. The profiles here are the merged version of
      the explicit and implicit profiles.
    geometry: Geometry at this time step used for the simulation.
    solver_numeric_outputs: Numerical quantities related to the solver.
  """

  t: jax.Array
  dt: jax.Array
  core_profiles: state.CoreProfiles
  core_transport: state.CoreTransport
  core_sources: source_profiles.SourceProfiles
  geometry: geometry.Geometry
  solver_numeric_outputs: state.SolverNumericOutputs

  def check_for_errors(self) -> state.SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.negative_temperature_or_density():
      logging.info("%s", self.core_profiles)
      _log_negative_profile_names(self.core_profiles)
      return state.SimError.NEGATIVE_CORE_PROFILES
    # If there are NaNs that occurred without negative core profiles, log this
    # as a separate error.
    if _has_nan(self):
      logging.info("%s", self.core_profiles)
      return state.SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return state.SimError.QUASINEUTRALITY_BROKEN
    else:
      return state.SimError.NO_ERROR


def _has_nan(
    inputs: ToraxSimState,
) -> bool:
  return any([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(inputs)])


def _log_negative_profile_names(inputs: state.CoreProfiles):
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  for path, value in path_vals:
    if jnp.any(jnp.less(value, 0.0)):
      logging.info("Found negative value in %s", jax.tree_util.keystr(path))
