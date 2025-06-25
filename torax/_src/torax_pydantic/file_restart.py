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

"""Pydantic config for restarting a simulation from a file."""
import pydantic

from torax._src.torax_pydantic import torax_pydantic


class FileRestart(torax_pydantic.BaseModelFrozen):
  """Pydantic config for restarting a simulation from a file.

  Attributes:
    filename: Filename to load initial state from.
    time: Time in state file at which to load from.
    do_restart: Toggle loading initial state from file or not.
    stitch: Whether to stitch the state from the file.
  """

  filename: pydantic.FilePath
  time: torax_pydantic.Second
  do_restart: bool
  stitch: bool
