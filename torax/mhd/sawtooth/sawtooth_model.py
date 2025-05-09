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
import functools
import jax
from torax import jax_utils
from torax import post_processing
from torax import state
from torax.config import runtime_params_slice
from torax.core_profiles import updaters
from torax.geometry import geometry
from torax.mhd.sawtooth import redistribution_base
from torax.mhd.sawtooth import trigger_base
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.transport_model import transport_model as transport_model_lib


# TODO(b/414537757). Sawtooth extensions.
# a. Full and incomplete Kadomtsev redistribution model.
# b. Porcelli model with free parameters and fast ion sensitivities.
# c. "Smooth" version that can work with forward-sensitivity-analysis and
#    stationary-state applications without the need for averaging.
class SawtoothModel:
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def __init__(
      self,
      trigger_model: trigger_base.TriggerModel,
      redistribution_model: redistribution_base.RedistributionModel,
      transport_model: transport_model_lib.TransportModel,
      pedestal_model: pedestal_model_lib.PedestalModel,
      source_models: source_models_lib.SourceModels,
  ):
    self._trigger_model = trigger_model
    self._redistribution_model = redistribution_model
    self._transport_model = transport_model
    self._pedestal_model = pedestal_model
    self._source_models = source_models

  @functools.partial(
      jax_utils.jit,
      static_argnames=[
          'self',
          'static_runtime_params_slice',
      ],
  )
  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      input_state: state.ToraxSimState,
      input_post_processed_outputs: state.PostProcessedOutputs,
      dynamic_runtime_params_slice_t_plus_crash_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t_plus_crash_dt: geometry.Geometry,
  ) -> tuple[state.ToraxSimState, state.PostProcessedOutputs]:
    """Applies the sawtooth model and outputs a new state if triggered.

    If the trigger model indicates a crash has been triggered, an
    instantaneous redistribution model is applied. A new state following a short
    (configurable) dt is returned, with core_profiles further modified by psidot
    assumed at time t, the new boundary conditions, and sources, transport, geo,
    and pedestal outputs consistent with the new core_profiles at time
    t_plus_crash_dt.

    Args:
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice_t: Dynamic runtime parameters at time t.
      input_state: The input ToraxSimState.
      input_post_processed_outputs: The input PostProcessedOutputs.
      dynamic_runtime_params_slice_t_plus_crash_dt: Dynamic runtime parameters
        at time t + crash_step_duration.
      geo_t_plus_crash_dt: The geometry at time t + crash_step_duration.

    Returns:
      The output ToraxSimState, which may be modified by the sawtooth model.
      The output PostProcessedOutputs, which may be modified by the sawtooth
        model.
    """

    trigger_sawtooth, rho_norm_q1 = self._trigger_model(
        static_runtime_params_slice,
        dynamic_runtime_params_slice_t,
        input_state.geometry,
        input_state.core_profiles,
    )

    def _make_redistributed_state_and_post_processed_outputs() -> (
        tuple[state.ToraxSimState, state.PostProcessedOutputs]
    ):
      # assertion needed for linter
      assert dynamic_runtime_params_slice_t.mhd.sawtooth is not None

      redistributed_core_profiles = self._redistribution_model(
          rho_norm_q1,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          input_state.core_profiles,
      )

      # Evolve the psi profile over the sawtooth time.
      # Redistribution maintains the same psi boundary condition. However,
      # over the course of the sawtooth time, the central solenoid must still
      # modify the psi profile. Since we don't calculate the psi PDE here, we
      # assume that for the short sawtooth time we can use the psidot from the
      # beginning of the step interval. This updates the bulk values. Later, the
      # boundary conditions are also updated at time t_plus_crash_dt when
      # using `updaters.update_all_core_profiles_after_step`.
      evolved_psi_redistributed_value = (
          redistributed_core_profiles.psi.value
          + input_state.core_profiles.psidot.value
          * dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration
      )
      evolved_core_profiles = dataclasses.replace(
          redistributed_core_profiles,
          psi=dataclasses.replace(
              redistributed_core_profiles.psi,
              value=evolved_psi_redistributed_value,
          ),
      )

      source_profiles = source_profile_builders.get_all_source_profiles(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          core_profiles=evolved_core_profiles,
          source_models=self._source_models,
      )

      # Needed to use update_all_core_profiles_after_step
      evolving_names = []
      if static_runtime_params_slice.evolve_ion_heat:
        evolving_names.append('temp_ion')
      if static_runtime_params_slice.evolve_electron_heat:
        evolving_names.append('temp_el')
      if static_runtime_params_slice.evolve_current:
        evolving_names.append('psi')
      if static_runtime_params_slice.evolve_density:
        evolving_names.append('n_e')
      evolving_names = tuple(evolving_names)

      x_new_redistributed = tuple(
          [getattr(evolved_core_profiles, name) for name in evolving_names]
      )

      # Prepare core_profiles_t_plus_crash_dt with new boundary conditions
      # and prescribed profiles if present.
      core_profiles_t_plus_crash_dt = updaters.provide_core_profiles_t_plus_dt(
          dt=dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_dt=geo_t_plus_crash_dt,
          core_profiles_t=input_state.core_profiles,
      )

      final_core_profiles = updaters.update_all_core_profiles_after_step(
          x_new_redistributed,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          source_profiles,
          input_state.core_profiles,
          core_profiles_t_plus_crash_dt,
          evolving_names,
          dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
      )

      pedestal_model_output = self._pedestal_model(
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          final_core_profiles,
      )

      transport_coeffs = self._transport_model(
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          final_core_profiles,
          pedestal_model_output,
      )

      output_state = dataclasses.replace(
          input_state,
          core_profiles=final_core_profiles,
          t=input_state.t
          + dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
          dt=dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
          geometry=geo_t_plus_crash_dt,
          core_transport=transport_coeffs,
          core_sources=source_profiles,
          sawtooth_crash=True,
      )
      output_post_processed_outputs = post_processing.make_post_processed_outputs(
          sim_state=output_state,
          dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_crash_dt,
          previous_post_processed_outputs=input_post_processed_outputs,
      )
      return output_state, output_post_processed_outputs

    output_state, output_post_processed_outputs = jax.lax.cond(
        trigger_sawtooth,
        _make_redistributed_state_and_post_processed_outputs,
        lambda: (
            dataclasses.replace(input_state, sawtooth_crash=False),
            input_post_processed_outputs,
        ),
    )

    return output_state, output_post_processed_outputs

  def __hash__(self) -> int:
    return hash((
        self._trigger_model,
        self._redistribution_model,
        self._transport_model,
        self._pedestal_model,
        self._source_models,
    ))

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, SawtoothModel)
        and self._trigger_model == other._trigger_model
        and self._redistribution_model == other._redistribution_model
        and self._transport_model == other._transport_model
        and self._pedestal_model == other._pedestal_model
        and self._source_models == other._source_models
    )
