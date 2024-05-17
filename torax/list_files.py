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
"""Contains the `list_files` function.

Lists all the files in torax.
"""
import os


def list_files(suffix: str = ""):
  """Returns a list of all files in torax with the given suffix.

  Args:
    suffix: str to match at the end of filenames.

  Returns:
    file_list: A list of all files in torax whose filepath ends with `suffix`
  """

  # Assumes lists_files.py is in the top of the module, need to adjust this
  # line if it is moved
  torax_path = os.path.dirname(__file__)

  file_list = _list_files(torax_path, suffix)

  return file_list


def _list_files(path: str, suffix: str = ""):
  """Recursive main loop helper function for `list_files`.

  Args:
    path: a filepath
    suffix : suffix to match


  Returns:
    l : A list of all files ending in `suffix` contained within `path`.
      (If `path` is a file rather than a directory, it is considered
      to "contain" itself)
  """
  if os.path.isdir(path):
    incomplete = os.listdir(path)
    complete = [os.path.join(path, entry) for entry in incomplete]
    lists = [_list_files(subpath, suffix) for subpath in complete]
    flattened = []
    for l in lists:
      for elem in l:
        flattened.append(elem)
    return flattened
  else:
    assert os.path.exists(path), "couldn't find file '%s'" % path
    if path.endswith(suffix):
      return [path]
    return []


if __name__ == "__main__":
  # Print all .py files in the library
  result = list_files(".py")
  for f in result:
    print(f)
