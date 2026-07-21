# Copyright 2026 DeepMind Technologies Limited
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

"""Identical to iterhybrid_predictor_corrector but with small fixed timestep."""
import copy

from torax.examples import iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
CONFIG['numerics']['fixed_dt'] = 1e-3
CONFIG['time_step_calculator'] = {'calculator_type': 'fixed'}
