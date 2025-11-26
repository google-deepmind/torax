
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
"""Useful functions for working with paths in TORAX."""
import pathlib


def torax_path() -> pathlib.Path:
  """Returns the absolute path to the Torax directory."""

  path = pathlib.Path(__file__).parent.parent
  assert path.is_dir(), f'Path {path} is not a directory.'
  assert path.parts[-1] == 'torax', f'Path {path} is not a Torax directory.'
  return path
