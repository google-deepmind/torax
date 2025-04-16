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
`build_static_runtime_params_slice`.
For the dynamic_runtime_params_slice this is a class
`DynamicRuntimeParamsSliceProvider` which provides a slice of the
DynamicRuntimeParamsSlice to use during time t of the sim.
This module also provides a method
`get_consistent_dynamic_runtime_params_slice_and_geometry` which returns a
DynamicRuntimeParamsSlice and a corresponding geometry with consistent Ip.
"""
import chex
from torax.config import numerics as numerics_lib
from torax.config import plasma_composition as plasma_composition_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params_lib
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.mhd import pydantic_model as mhd_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.torax_pydantic import model_config
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import pydantic_model as transport_pydantic_model
from torax.transport_model import pydantic_model_base as base_transport_pydantic_model
import typing_extensions


def build_static_params_from_config(
    config: model_config.ToraxConfig,
) -> runtime_params_slice.StaticRuntimeParamsSlice:
  """Builds a StaticRuntimeParamsSlice from a ToraxConfig."""
  return build_static_runtime_params_slice(
      profile_conditions=config.runtime_params.profile_conditions,
      numerics=config.runtime_params.numerics,
      plasma_composition=config.runtime_params.plasma_composition,
      sources=config.sources,
      torax_mesh=config.geometry.build_provider.torax_mesh,
      stepper=config.stepper,
  )


def build_static_runtime_params_slice(
    *,
    profile_conditions: profile_conditions_lib.ProfileConditions,
    numerics: numerics_lib.Numerics,
    plasma_composition: plasma_composition_lib.PlasmaComposition,
    sources: sources_pydantic_model.Sources,
    torax_mesh: torax_pydantic.Grid1D,
    stepper: stepper_pydantic_model.Stepper | None = None,
) -> runtime_params_slice.StaticRuntimeParamsSlice:
  """Builds a StaticRuntimeParamsSlice.

  Args:
    profile_conditions: Profile conditions from which the profile conditions
      static variables are taken, which are the boundary conditions for the
      plasma.
    numerics: Numerics from which the numerics static variables are taken, which
      are the equations being solved, adaptive dt, and the fixed dt.
    plasma_composition: Plasma composition from which the plasma composition
      static variables are taken, which are the main and impurity ion names.
    sources: data from which the source related static variables are taken,
      which are the explicit/implicit toggle and calculation mode for each
      source.
    torax_mesh: The torax mesh, e.g. the grid used to construct the geometry.
      This is static for the entire simulation and any modification implies
      changed array sizes, and hence would require a recompilation. Useful to
      have a static (concrete) mesh for various internal calculations.
    stepper: stepper runtime params from which stepper static variables are
      extracted, related to solver methods. If None, defaults to the default
      stepper runtime params.

  Returns:
    A runtime_params_slice.StaticRuntimeParamsSlice.
  """
  stepper = stepper or stepper_pydantic_model.Stepper()
  return runtime_params_slice.StaticRuntimeParamsSlice(
      sources={
          source_name: source_config.build_static_params()
          for source_name, source_config in sources.source_model_config.items()
      },
      torax_mesh=torax_mesh,
      stepper=stepper.build_static_params(),
      ion_heat_eq=numerics.ion_heat_eq,
      el_heat_eq=numerics.el_heat_eq,
      current_eq=numerics.current_eq,
      dens_eq=numerics.dens_eq,
      main_ion_names=plasma_composition.get_main_ion_names(),
      impurity_names=plasma_composition.get_impurity_names(),
      adaptive_dt=numerics.adaptive_dt,
      use_vloop_lcfs_boundary_condition=profile_conditions.use_vloop_lcfs_boundary_condition,
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

  In more detail if you are updating any interpolated variable constructors
  (e.g. `runtime_params.profile_conditions.Ti_bound_right`) you will need to
  construct a new provider object. If you are only updating static variables
  (e.g. `runtime_params.profile_conditions.normalize_to_nbar`) then you can
  update the runtime params object in place and the changes will be picked up in
  the provider you have.

  For example to update the Ti_bound_right interpolated var constructor:
  ```
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  provider = DynamicRuntimeParamsSliceProvider(
      runtime_params=runtime_params,
      torax_mesh=torax_mesh,
  )
  runtime_params.profile_conditions.Ti_bound_right = new_value
  provider = DynamicRuntimeParamsSliceProvider(
      runtime_params=runtime_params,
      torax_mesh=torax_mesh,
  )
  ```
  """

  def __init__(
      self,
      runtime_params: general_runtime_params_lib.GeneralRuntimeParams,
      pedestal: pedestal_pydantic_model.BasePedestal | None = None,
      transport: base_transport_pydantic_model.TransportBase | None = None,
      sources: sources_pydantic_model.Sources | None = None,
      stepper: stepper_pydantic_model.Stepper | None = None,
      mhd: mhd_pydantic_model.MHD | None = None,
      torax_mesh: torax_pydantic.Grid1D | None = None,
  ):
    """Constructs a build_simulation_params.DynamicRuntimeParamsSliceProvider.

    Args:
      runtime_params: The general runtime params to use.
      pedestal: The pedestal model runtime params to use. If None, defaults to
        the default pedestal model runtime params.
      transport: The transport model runtime params to use. If None, defaults to
        the default transport model runtime params.
      sources: A dict of source name to source runtime params to use. If None,
        defaults to an empty dict (i.e. no sources).
      stepper: The stepper configuration to use. If None, defaults to the
        default stepper configuration.
      mhd: The mhd configuration to use. If None, defaults to an empty MHD
        object.
      torax_mesh: The torax mesh to use. If the slice provider doesn't need to
        construct any rho interpolated values, this can be None, else an error
        will be raised within the constructor of the interpolated variable.
    """
    torax_pydantic.set_grid(runtime_params, torax_mesh, mode='relaxed')
    transport = transport or transport_pydantic_model.QLKNNTransportModel()
    sources = sources or sources_pydantic_model.Sources()
    stepper = stepper or stepper_pydantic_model.Stepper()
    pedestal = pedestal or pedestal_pydantic_model.NoPedestal()
    mhd = mhd or mhd_pydantic_model.MHD()
    torax_pydantic.set_grid(sources, torax_mesh, mode='relaxed')
    self._torax_mesh = torax_mesh
    self._sources = sources
    self._runtime_params = runtime_params
    self._transport_model = transport
    self._stepper = stepper
    self._pedestal = pedestal
    self._mhd = mhd

  @classmethod
  def from_config(
      cls,
      config: model_config.ToraxConfig,
  ) -> typing_extensions.Self:
    """Constructs a DynamicRuntimeParamsSliceProvider from a ToraxConfig."""
    return cls(
        runtime_params=config.runtime_params,
        pedestal=config.pedestal,
        transport=config.transport,
        sources=config.sources,
        stepper=config.stepper,
        mhd=config.mhd,
        torax_mesh=config.geometry.build_provider.torax_mesh,
    )

  @property
  def sources(self) -> sources_pydantic_model.Sources:
    return self._sources

  def validate_new(
      self,
      new_provider: typing_extensions.Self,
  ):
    """Validates that the new provider is compatible."""
    if set(new_provider.sources.source_model_config.keys()) != set(
        self.sources.source_model_config.keys()
    ):
      raise ValueError(
          'New dynamic runtime params slice provider has different sources.'
      )

  def __call__(
      self,
      t: chex.Numeric,
  ) -> runtime_params_slice.DynamicRuntimeParamsSlice:
    """Returns a runtime_params_slice.DynamicRuntimeParamsSlice to use during time t of the sim."""
    dynamic_general_runtime_params = self._runtime_params.build_dynamic_params(
        t
    )
    return runtime_params_slice.DynamicRuntimeParamsSlice(
        transport=self._transport_model.build_dynamic_params(t),
        stepper=self._stepper.build_dynamic_params,
        sources={
            source_name: input_source_config.build_dynamic_params(t)
            for source_name, input_source_config in self._sources.source_model_config.items()
        },
        plasma_composition=dynamic_general_runtime_params.plasma_composition,
        profile_conditions=dynamic_general_runtime_params.profile_conditions,
        numerics=dynamic_general_runtime_params.numerics,
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
