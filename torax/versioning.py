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
"""Functionality for fine-grained versioning of torax.

Contains `torax_hash` and the functionality to calculate it.
`torax_hash` should be used in the __hash__ and __eq__ method of all classes
used as arguments to jitted Jax functions.
This ensures that changes to Torax itself cause new code with the new
Torax behavior to be compiled and executed, rather than old cached code with
the old behavior to be executed.
"""
from torax import list_files


def get_file_contents(path: str) -> str:
  """Returns the contents of a file.

  Delimited to make sure there isn't ambiguity when two files are concatenated
  one after another.

  Args:
    path: Path to the file.

  Returns:
    contents: The contents of the file, delimited to prevent ambiguity about
    where the content came from when multiple files are concatenated next to
    each other..
  """
  with open(path, 'r') as f:
    contents = f.read()
  delimited = f'<file path="{path}">{contents}</file>'
  return delimited


def calc_torax_hash() -> int:
  """Hashes all of the python code in the entire Torax module."""
  files = list_files.list_files('.py')
  contents = [get_file_contents(f) for f in files]
  module_contents = ','.join(contents)
  return hash(module_contents)


# We calculate this once on import so it can be reused many times, otherwise
# hash functions of all the torax classes would be slow.
torax_hash = calc_torax_hash()
