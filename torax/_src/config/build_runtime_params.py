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

For the static_runtime_params_slice this is a method
`build_static_runtime_params__from_config`.
For the dynamic_runtime_params_slice this is a class
`DynamicRuntimeParamsSliceProvider` which provides a slice of the
DynamicRuntimeParamsSlice to use during time t of the sim.
This module also provides a method
`get_consistent_dynamic_runtime_params_slice_and_geometry` which returns a
DynamicRuntimeParamsSlice and a corresponding geometry with consistent Ip.
"""
import chex
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.torax_pydantic import model_config
import typing_extensions


def build_static_params_from_config(
    config: model_config.ToraxConfig,
) -> runtime_params_slice.StaticRuntimeParamsSlice:
  """Builds a StaticRuntimeParamsSlice from a ToraxConfig."""
  return runtime_params_slice.StaticRuntimeParamsSlice(
      sources={
          source_name: source_config.build_static_params()
          for source_name, source_config in dict(config.sources).items()
          if source_config is not None
      },
      torax_mesh=config.geometry.build_provider.torax_mesh,
      solver=config.solver.build_static_params(),
      main_ion_names=config.plasma_composition.get_main_ion_names(),
      impurity_names=config.plasma_composition.get_impurity_names(),
      numerics=config.numerics.build_static_params(),
      profile_conditions=config.profile_conditions.build_static_params(),
  )


class DynamicRuntimeParamsSliceProvider:
  """Provides a DynamicRuntimeParamsSlice to use during time t of the sim.

  The DynamicRuntimeParamsSlice may change from time step to time step, so this
  class interpolates any time-dependent params in the input config to the values
  they should be at time t.

  NOTE: In order to maintain consistency between the DynamicRuntimeParamsSlice
  and the geometry,
  `sim.get_consistent_dynamic_runtime_params_slice_and_geometry`
  should be used to get a slice of the DynamicRuntimeParamsSlice and a
  corresponding geometry.

  See `run_simulation()` for how this callable is used.

  After this object has been constructed changes any runtime params may not
  be picked up if they are updated and it is safest to construct a new provider
  object (if for example updating the simulation).
  ```
  """

  def __init__(
      self,
      torax_config: model_config.ToraxConfig,
  ):
    """Constructs a build_simulation_params.DynamicRuntimeParamsSliceProvider."""
    self._sources = torax_config.sources
    self._numerics = torax_config.numerics
    self._profile_conditions = torax_config.profile_conditions
    self._plasma_composition = torax_config.plasma_composition
    self._transport_model = torax_config.transport
    self._solver = torax_config.solver
    self._pedestal = torax_config.pedestal
    self._mhd = torax_config.mhd
    self._neoclassical = torax_config.neoclassical

  @property
  def numerics(self) -> numerics_lib.Numerics:
    return self._numerics

  @classmethod
  def from_config(
      cls,
      config: model_config.ToraxConfig,
  ) -> typing_extensions.Self:
    """Constructs a DynamicRuntimeParamsSliceProvider from a ToraxConfig."""
    return cls(config)

  def __call__(
      self,
      t: chex.Numeric,
  ) -> runtime_params_slice.DynamicRuntimeParamsSlice:
    """Returns a runtime_params_slice.DynamicRuntimeParamsSlice to use during time t of the sim."""
    return runtime_params_slice.DynamicRuntimeParamsSlice(
        transport=self._transport_model.build_dynamic_params(t),
        solver=self._solver.build_dynamic_params,
        sources={
            source_name: source_config.build_dynamic_params(t)
            for source_name, source_config in dict(self._sources).items()
            if source_config is not None
        },
        plasma_composition=self._plasma_composition.build_dynamic_params(t),
        profile_conditions=self._profile_conditions.build_dynamic_params(t),
        numerics=self._numerics.build_dynamic_params(t),
        neoclassical=self._neoclassical.build_dynamic_params(),
        pedestal=self._pedestal.build_dynamic_params(t),
        mhd=self._mhd.build_dynamic_params(t),
    )


def get_consistent_dynamic_runtime_params_slice_and_geometry(
    *,
    t: chex.Numeric,
    dynamic_runtime_params_slice_provider: DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[runtime_params_slice.DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Returns the dynamic runtime params and geometry for a given time."""
  geo = geometry_provider(t)
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      t=t,
  )
  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )
  return dynamic_runtime_params_slice, geo
