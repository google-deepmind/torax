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

"""Sawtooth model."""

import dataclasses
import jax
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.mhd.sawtooth import redistribution_base
from torax.mhd.sawtooth import trigger_base
from torax.neoclassical.conductivity import base as base_conductivity
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib
from torax.stepper import stepper
from torax.transport_model import transport_model as transport_model_lib


# TODO(b/414537757). Sawtooth extensions.
# a. Full and incomplete Kadomtsev redistribution model.
# b. Porcelli model with free parameters and fast ion sensitivities.
# c. "Smooth" version that can work with forward-sensitivity-analysis and
#    stationary-state applications without the need for averaging.
class SawtoothModel(stepper.Solver):
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def __init__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      trigger_model: trigger_base.TriggerModel,
      redistribution_model: redistribution_base.RedistributionModel,
      transport_model: transport_model_lib.TransportModel,
      pedestal_model: pedestal_model_lib.PedestalModel,
      source_models: source_models_lib.SourceModels,
  ):
    super().__init__(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )
    self._trigger_model = trigger_model
    self._redistribution_model = redistribution_model

  def _x_new(
      self,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      core_sources_t: source_profiles_lib.SourceProfiles,
      core_transport_t: state.CoreTransport,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      source_profiles_lib.SourceProfiles,
      base_conductivity.Conductivity,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """Applies the sawtooth model and outputs new state attributes if triggered.

    If the trigger model indicates a crash has been triggered, an
    instantaneous redistribution model is applied. New state attributes
    following a short (configurable) dt are returned. Beyond the sawtooth
    redistribution, core_profiles are further updated by: the psidot assumed at
    time t; the new boundary conditions; sources, transport, geo, and pedestal
    outputs consistent with the new core_profiles at time t_plus_crash_dt.

    Args:
      dt: Sawtooth step duration.
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice_t: Dynamic runtime parameters at time t.
      dynamic_runtime_params_slice_t_plus_dt: Dynamic runtime parameters at time
        t + crash_dt.
      geo_t: Geometry at time t.
      geo_t_plus_dt: Geometry at time t + crash_dt.
      core_profiles_t: Core profiles at time t.
      core_profiles_t_plus_dt: Core profiles containing boundary conditions and
        prescribed profiles at time t + crash_dt.
      core_sources_t: Source profiles at time t.
      core_transport_t: Transport coefficients at time t.
      explicit_source_profiles: Explicit source profiles at time t.
      evolving_names: Names of evolving variables.

    Returns:
      Updated tuple of evolving CellVariables from CoreProfiles
      Source profiles consistent with redistributed state.
      Conductivity consistent with redistributed state.
      Transport coefficients consistent with redistributed state.
      SolverNumericOutputs indicating a sawtooth crash.
    """

    trigger_sawtooth, rho_norm_q1 = self._trigger_model(
        static_runtime_params_slice,
        dynamic_runtime_params_slice_t,
        geo_t,
        core_profiles_t,
    )

    def _redistribute_state() -> tuple[
        tuple[cell_variable.CellVariable, ...],
        source_profiles_lib.SourceProfiles,
        base_conductivity.Conductivity,
        state.CoreTransport,
        state.SolverNumericOutputs,
    ]:

      redistributed_core_profiles = self._redistribution_model(
          rho_norm_q1,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t,
          geo_t,
          core_profiles_t,
      )

      # Evolve the psi profile over the sawtooth time.
      # Redistribution maintains the same psi boundary condition. However,
      # over the course of the sawtooth time, the central solenoid must still
      # modify the psi profile. Since we don't calculate the psi PDE here, we
      # assume that for the short sawtooth time we can use the psidot from the
      # beginning of the step interval. This updates the bulk values. Later, the
      # boundary conditions are also updated at time t_plus_dt when
      # using `updaters.update_all_core_profiles_after_step`.
      evolved_psi_redistributed_value = (
          redistributed_core_profiles.psi.value
          + core_profiles_t.psidot.value * dt
      )
      evolved_core_profiles = dataclasses.replace(
          redistributed_core_profiles,
          psi=dataclasses.replace(
              redistributed_core_profiles.psi,
              value=evolved_psi_redistributed_value,
          ),
      )

      conductivity_post_step = (
          self.source_models.conductivity.calculate_conductivity(
              dynamic_runtime_params_slice_t_plus_dt,
              geo_t_plus_dt,
              redistributed_core_profiles,
          )
      )
      core_sources_post_step = source_profile_builders.get_all_source_profiles(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
          core_profiles=evolved_core_profiles,
          source_models=self.source_models,
          conductivity=conductivity_post_step,
      )

      x_post_step = tuple(
          [getattr(evolved_core_profiles, name) for name in evolving_names]
      )

      pedestal_model_output = self.pedestal_model(
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
          redistributed_core_profiles,
      )

      core_transport_post_step = self.transport_model(
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
          redistributed_core_profiles,
          pedestal_model_output,
      )

      solver_numeric_outputs_post_step = state.SolverNumericOutputs(
          sawtooth_crash=True
      )

      return (
          x_post_step,
          core_sources_post_step,
          conductivity_post_step,
          core_transport_post_step,
          solver_numeric_outputs_post_step,
      )

    # Return redistributed state attributes if triggered, otherwise return
    # unchanged state attributes.
    return jax.lax.cond(
        trigger_sawtooth,
        _redistribute_state,
        lambda: (
            tuple([getattr(core_profiles_t, name) for name in evolving_names]),
            core_sources_t,
            base_conductivity.Conductivity(
                sigma=core_profiles_t.sigma,
                sigma_face=core_profiles_t.sigma_face),
            core_transport_t,
            state.SolverNumericOutputs(),
        ),
    )

  def __hash__(self) -> int:
    return hash((
        self._trigger_model,
        self._redistribution_model,
        self.static_runtime_params_slice,
        self.transport_model,
        self.pedestal_model,
        self.source_models,
    ))

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, SawtoothModel)
        and self._trigger_model == other._trigger_model
        and self._redistribution_model == other._redistribution_model
        and self.static_runtime_params_slice
        == other.static_runtime_params_slice
        and self.transport_model == other.transport_model
        and self.pedestal_model == other.pedestal_model
        and self.source_models == other.source_models
    )
