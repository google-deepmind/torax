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
"""Methods for building simulation parameters.

 - `RuntimeParamsProvider` which provides a the `RuntimeParams` to
  use during time t of the sim.
 - `get_consistent_params_and_geometry` which returns a
`RuntimeParams` and a corresponding `Geometry` with consistent `Ip`.
"""
import dataclasses

import chex
import jax
from torax._src import jax_utils
from torax._src.config import numerics as numerics_lib
from torax._src.config import plasma_composition as plasma_composition_lib
from torax._src.config import profile_conditions as profile_conditions_lib
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model as transport_pydantic_model
import typing_extensions


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
  """Provides a RuntimeParamsSlice to use during time t of the sim.

  The RuntimeParams may change from time step to time step, so this
  class interpolates any time-dependent params in the input config to the values
  they should be at time t.

  NOTE: In order to maintain consistency between the RuntimeParams
  and the geometry,
  `sim.get_consistent_dynamic_runtime_params_slice_and_geometry`
  should be used to get a slice of the RuntimeParams and a
  corresponding geometry. See `run_simulation()` for how this callable is used.
  """

  sources: sources_pydantic_model.Sources
  numerics: numerics_lib.Numerics
  profile_conditions: profile_conditions_lib.ProfileConditions
  plasma_composition: plasma_composition_lib.PlasmaComposition
  transport_model: transport_pydantic_model.TransportConfig
  solver: solver_pydantic_model.SolverConfig
  pedestal: pedestal_pydantic_model.PedestalConfig
  mhd: mhd_pydantic_model.MHD
  neoclassical: neoclassical_pydantic_model.Neoclassical
  time_step_calculator: time_step_calculator_pydantic_model.TimeStepCalculator

  @classmethod
  def from_config(
      cls,
      config: model_config.ToraxConfig,
  ) -> typing_extensions.Self:
    """Constructs a DynamicRuntimeParamsSliceProvider from a ToraxConfig."""
    return cls(
        sources=config.sources,
        numerics=config.numerics,
        profile_conditions=config.profile_conditions,
        plasma_composition=config.plasma_composition,
        transport_model=config.transport,
        solver=config.solver,
        pedestal=config.pedestal,
        mhd=config.mhd,
        neoclassical=config.neoclassical,
        time_step_calculator=config.time_step_calculator,
    )

  @jax_utils.jit
  def __call__(
      self,
      t: chex.Numeric,
  ) -> runtime_params_slice.RuntimeParams:
    """Returns a runtime_params_slice.RuntimeParams to use during time t of the sim."""
    return runtime_params_slice.RuntimeParams(
        transport=self.transport_model.build_runtime_params(t),
        solver=self.solver.build_dynamic_params,
        sources={
            source_name: source_config.build_dynamic_params(t)
            for source_name, source_config in dict(self.sources).items()
            if source_config is not None
        },
        plasma_composition=self.plasma_composition.build_dynamic_params(t),
        profile_conditions=self.profile_conditions.build_dynamic_params(t),
        numerics=self.numerics.build_dynamic_params(t),
        neoclassical=self.neoclassical.build_dynamic_params(),
        pedestal=self.pedestal.build_dynamic_params(t),
        mhd=self.mhd.build_dynamic_params(t),
        time_step_calculator=self.time_step_calculator.build_dynamic_params(),
    )


def get_consistent_runtime_params_and_geometry(
    *,
    t: chex.Numeric,
    runtime_params_provider: RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[runtime_params_slice.RuntimeParams, geometry.Geometry]:
  """Returns the runtime params and geometry for a given time."""
  geo = geometry_provider(t)
  runtime_params = runtime_params_provider(t=t)
  return runtime_params_slice.make_ip_consistent(runtime_params, geo)
