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


"""Base and analytical classes for Neoclassical models."""
import abc
import dataclasses
import jax
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base as poloidal_velocity_base
from torax._src.neoclassical.transport import base as transport_base
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NeoclassicalOutputs:
  """Consolidated outputs from a neoclassical model evaluation."""

  conductivity: conductivity_base.Conductivity
  bootstrap_current: bootstrap_current_base.BootstrapCurrent
  transport: transport_coeffs_lib.NeoclassicalTransport
  poloidal_velocity: poloidal_velocity_base.PoloidalVelocity


@dataclasses.dataclass(frozen=True, eq=False)
class NeoclassicalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for neoclassical models.

  A neoclassical model computes all neoclassical quantities (conductivity,
  bootstrap current, neoclassical transport, and poloidal velocity) and returns
  a `NeoclassicalOutputs` instance when called. Subclasses can either be
  composite models that delegate to individual analytical sub-models (see
  `AnalyticalNeoclassicalModel`) or integrated neoclassical solvers that compute
  all outputs in a single evaluation.
  """

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> NeoclassicalOutputs:
    """Evaluates the neoclassical model at the given state."""


@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalNeoclassicalModel(NeoclassicalModel):
  """Composite neoclassical model delegating to analytical sub-models.

  Evaluates individual analytical sub-models for conductivity, bootstrap
  current, neoclassical transport, and poloidal velocity in a single pass,
  sharing precomputed intermediate quantities via `AnalyticalCache`.
  """

  conductivity: conductivity_base.ConductivityModel
  bootstrap_current: bootstrap_current_base.BootstrapCurrentModel
  transport: transport_base.NeoclassicalTransportModel
  poloidal_velocity: poloidal_velocity_base.PoloidalVelocityModel

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> NeoclassicalOutputs:
    """Evaluates all analytical neoclassical sub-models in a single pass."""
    analytical_cache = formulas.compute_analytical_cache(geo, core_profiles)
    conductivity = self.conductivity.calculate_conductivity(
        geo, core_profiles, analytical_cache=analytical_cache
    )
    bootstrap_current = self.bootstrap_current.calculate_bootstrap_current(
        runtime_params, geo, core_profiles, analytical_cache=analytical_cache
    )
    transport = self.transport(
        runtime_params, geo, core_profiles, analytical_cache=analytical_cache
    )
    poloidal_velocity = self.poloidal_velocity.calculate_poloidal_velocity(
        runtime_params, geo, core_profiles, analytical_cache=analytical_cache
    )
    return NeoclassicalOutputs(
        conductivity=conductivity,
        bootstrap_current=bootstrap_current,
        transport=transport,
        poloidal_velocity=poloidal_velocity,
    )
