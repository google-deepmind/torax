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
import dataclasses

from absl import logging
import jax
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.edge import base as edge_base
from torax._src.geometry import geometry
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
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
    edge_outputs: Outputs from the edge model, if one is active.
    geometry: Geometry at this time step used for the simulation.
    solver_numeric_outputs: Numerical quantities related to the solver.
  """

  t: array_typing.FloatScalar
  dt: array_typing.FloatScalar
  core_profiles: state.CoreProfiles
  core_transport: state.CoreTransport
  core_sources: source_profiles.SourceProfiles
  edge_outputs: edge_base.EdgeModelOutputs | None
  geometry: geometry.Geometry
  solver_numeric_outputs: state.SolverNumericOutputs

  def check_for_errors(self) -> state.SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.negative_temperature_or_density():
      logging.info("Unphysical negative values detected in core profiles:\n")
      _log_negative_profile_names(self.core_profiles)
      return state.SimError.NEGATIVE_CORE_PROFILES
    if self.has_nan():
      logging.info("NaNs detected in ToraxSimState:\n")
      _log_nans(self)
      return state.SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return state.SimError.QUASINEUTRALITY_BROKEN
    else:
      return state.SimError.NO_ERROR

  def has_nan(self) -> bool:
    return any([np.any(np.isnan(x)) for x in jax.tree.leaves(self)])


def _log_nans(
    inputs: ToraxSimState,
) -> None:
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  nan_count = 0
  for path, value in path_vals:
    if np.any(np.isnan(value)):
      logging.info("Found NaNs in sim_state%s", jax.tree_util.keystr(path))
      nan_count += 1
  if nan_count >= 10:
    logging.info("""\nA common cause of widespread NaNs is negative densities or
        temperatures evolving during the solver step. This often arises through
        physical reasons like radiation collapse, or unphysical configuration
        such as impurity densities incompatible with physical quasineutrality.
        Check the output file for near-zero temperatures or densities at the
        last valid step.""")


def _log_negative_profile_names(inputs: state.CoreProfiles) -> None:
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  for path, value in path_vals:
    if np.any(np.less(value, 0.0)):
      logging.info("Found negative value in %s", jax.tree_util.keystr(path))
