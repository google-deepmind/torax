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
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical import runtime_params as neoclassical_runtime_params
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base as poloidal_velocity_base
from torax._src.neoclassical.transport import base as transport_base
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib


@dataclasses.dataclass(frozen=True, eq=False)
class NeoclassicalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for neoclassical models.

  Neoclassical model evaluations are split into two steps. First,
  `calculate_conductivity_current_and_rotation` computes parallel conductivity,
  bootstrap current, poloidal velocity, and optional model-specific
  intermediates (such as collisionalities and trapped-particle fractions for
  analytical models, or drift-kinetic moments for an integrated solver). Second,
  `calculate_transport` computes `NeoclassicalTransport` once `psidot` is
  available.
  """

  @abc.abstractmethod
  def calculate_conductivity_current_and_rotation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[
      conductivity_base.Conductivity,
      bootstrap_current_base.BootstrapCurrent,
      poloidal_velocity_base.PoloidalVelocity,
      neoclassical_intermediates_lib.NeoclassicalIntermediates,
  ]:
    """Calculates current, poloidal velocity, and intermediates."""

  @abc.abstractmethod
  def calculate_transport(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      intermediates: neoclassical_intermediates_lib.NeoclassicalIntermediates,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    """Calculates neoclassical transport coefficients."""


@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalNeoclassicalModel(NeoclassicalModel):
  """Composite neoclassical model delegating to analytical sub-models.

  Evaluates individual analytical sub-models for conductivity, bootstrap
  current, poloidal velocity, and neoclassical transport, sharing precomputed
  intermediate quantities via `AnalyticalNeoclassicalIntermediates`.
  """

  conductivity: conductivity_base.ConductivityModel
  bootstrap_current: bootstrap_current_base.BootstrapCurrentModel
  transport: transport_base.NeoclassicalTransportModel
  poloidal_velocity: poloidal_velocity_base.PoloidalVelocityModel

  def calculate_conductivity_current_and_rotation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[
      conductivity_base.Conductivity,
      bootstrap_current_base.BootstrapCurrent,
      poloidal_velocity_base.PoloidalVelocity,
      formulas.AnalyticalNeoclassicalIntermediates,
  ]:
    """Calculates current, poloidal velocity, and intermediates."""
    assert isinstance(
        runtime_params.neoclassical,
        neoclassical_runtime_params.AnalyticalRuntimeParams,
    )
    analytical_params = runtime_params.neoclassical
    neoclassical_intermediates = formulas.compute_neoclassical_intermediates(
        geo, core_profiles
    )
    conductivity = self.conductivity.calculate_conductivity(
        analytical_params.conductivity,
        geo,
        core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    bootstrap_current = self.bootstrap_current.calculate_bootstrap_current(
        analytical_params.bootstrap_current,
        geo,
        core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    poloidal_velocity = self.poloidal_velocity.calculate_poloidal_velocity(
        analytical_params.poloidal_velocity,
        geo,
        core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    return (
        conductivity,
        bootstrap_current,
        poloidal_velocity,
        neoclassical_intermediates,
    )

  def calculate_transport(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      intermediates: neoclassical_intermediates_lib.NeoclassicalIntermediates,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    """Calculates neoclassical transport coefficients."""
    assert isinstance(
        runtime_params.neoclassical,
        neoclassical_runtime_params.AnalyticalRuntimeParams,
    )
    assert isinstance(
        intermediates, formulas.AnalyticalNeoclassicalIntermediates
    )
    return self.transport(
        runtime_params.neoclassical.transport,
        geo,
        core_profiles,
        neoclassical_intermediates=intermediates,
    )
