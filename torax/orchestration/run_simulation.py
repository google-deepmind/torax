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
"""Contains the main programmatic entry point for running a TORAX simulation.

The intended use is
```
torax_config = torax.ToraxConfig.from_dict(config_dict)
sim_outputs = torax.run_simulation(torax_config)

# Update the config to run a new simulation with different parameters.
torax_config.update(updated_fields)
new_sim_outputs = torax.run_simulation(torax_config)
```
"""

from torax import output, sim
from torax.config import build_runtime_params
from torax.orchestration import initial_state as initial_state_lib
from torax.orchestration import step_function
from torax.sources import source_models as source_models_lib
from torax.torax_pydantic import model_config


def prep_simulation(
    torax_config: model_config.ToraxConfig,
) -> output.StateHistory:
    """Prep a TORAX simulation using the config and returns the necessary
    inputs for a simulation."""
    # TODO(b/384767453): Remove the need for the step_fn and stepper to take
    # the transport model and pedestal model.
    transport_model = torax_config.transport.build_transport_model()
    pedestal_model = torax_config.pedestal.build_pedestal_model()

    geometry_provider = torax_config.geometry.build_provider
    source_models = source_models_lib.SourceModels(
        torax_config.sources.source_model_config
    )

    stepper = torax_config.stepper.build_stepper_model(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )

    mhd_models = torax_config.mhd.build_mhd_models()

    step_fn = step_function.SimulationStepFn(
        stepper=stepper,
        time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
        transport_model=transport_model,
        pedestal_model=pedestal_model,
        mhd_models=mhd_models,
    )

    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=torax_config.runtime_params,
            sources=torax_config.sources,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
            stepper=torax_config.stepper,
        )
    )

    dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=torax_config.runtime_params,
            pedestal=torax_config.pedestal,
            transport=torax_config.transport,
            sources=torax_config.sources,
            stepper=torax_config.stepper,
            mhd=torax_config.mhd,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
        )
    )

    dynamic_runtime_params_slice_for_init, geo_for_init = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.runtime_params.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=geometry_provider,
        )
    )

    if torax_config.restart and torax_config.restart.do_restart:
        initial_state = initial_state_lib.initial_state_from_file_restart(
            file_restart=torax_config.restart,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_for_init=dynamic_runtime_params_slice_for_init,
            geo_for_init=geo_for_init,
            step_fn=step_fn,
        )
        restart_case = True
    else:
        initial_state = initial_state_lib.get_initial_state(
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
            geo=geo_for_init,
            step_fn=step_fn,
        )
        restart_case = False

    return (
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geometry_provider,
        initial_state,
        restart_case,
        step_fn,
        source_models,
    )


def run_simulation(
    torax_config: model_config.ToraxConfig,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> output.StateHistory:
    """Runs a TORAX simulation using the config and returns the outputs."""
    (
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geometry_provider,
        initial_state,
        restart_case,
        step_fn,
        source_models,
    ) = prep_simulation(torax_config)

    state_history, sim_error = sim._run_simulation(  # pylint: disable=protected-access
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
        geometry_provider=geometry_provider,
        initial_state=initial_state,
        restart_case=restart_case,
        step_fn=step_fn,
        log_timestep_info=log_timestep_info,
        progress_bar=progress_bar,
    )

    return output.StateHistory(
        state_history=state_history,
        sim_error=sim_error,
        source_models=source_models,
        torax_config=torax_config,
    )
