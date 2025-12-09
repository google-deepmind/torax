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

"""Experimental library functionality for TORAX."""
# pylint: disable=g-importing-member
from torax._src.config.build_runtime_params import RuntimeParamsProvider
from torax._src.config.build_runtime_params import ValidUpdates
from torax._src.orchestration.initial_state import get_initial_state_and_post_processed_outputs
from torax._src.orchestration.jit_run_loop import run_loop_jit
from torax._src.orchestration.run_simulation import make_step_fn
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.orchestration.sim_state import SimState
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic.interpolated_param_1d import TimeVaryingScalarUpdate
from torax._src.torax_pydantic.interpolated_param_2d import TimeVaryingArrayUpdate
from torax.experimental import geometry


__all__ = [
    'geometry',
    'make_step_fn',
    'prepare_simulation',
    'run_loop_jit',
    'RuntimeParamsProvider',
    'ValidUpdates',
    'SimulationStepFn',
    'TimeVaryingScalarUpdate',
    'TimeVaryingArrayUpdate',
    'get_initial_state_and_post_processed_outputs',
    'SimState',
]
