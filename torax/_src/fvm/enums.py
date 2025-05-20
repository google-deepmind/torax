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
"""The `enums` module.

Enums shared through the `fvm` package.
"""
import enum


@enum.unique
class InitialGuessMode(enum.Enum):
  """Modes for initial guess of x_new for iterative solvers."""

  # Initialize x_new = x_old
  X_OLD = 0

  # Use the linear solver to guess x_new
  LINEAR = 1
