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
    filename: Path to an existing TORAX output file (netCDF format).
    time: Time in output file to load as the new initial state.
    do_restart: If True, perform the restart from the selected state.
      If False, disable the restart and run the simulation as normal.
    stitch: If True, concatenate the old and new simulation histories
      in the resulting output file. If False, output file will only
      contain the new history.
  """

  filename: pydantic.FilePath
  time: torax_pydantic.Second
  do_restart: bool
  stitch: bool
