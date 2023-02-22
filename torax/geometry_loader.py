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

"""File I/O for loading geometry files."""

import jax.numpy as jnp


def initialize_CHEASE_dict(  # pylint: disable=invalid-name
    file_path: str,
) -> dict[str, jnp.ndarray]:
  """Loads the data from a CHEASE file into a dictionary."""
  # pyformat: disable
  with open(file_path, 'r') as file:
  # pyformat: enable
    chease_data = {}
    var_labels = file.readline().strip().split()[1:]  # ignore % comment column

    for var_label in var_labels:
      chease_data[var_label] = []

    # store data in respective keys
    for line in file:
      values = line.strip().split()
      for var_label, value in zip(var_labels, values):
        chease_data[var_label].append(float(value))

  # Convert lists to jax arrays.
  return {
      var_label: jnp.array(chease_data[var_label]) for var_label in chease_data
  }
