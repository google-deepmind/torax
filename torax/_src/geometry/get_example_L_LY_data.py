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

"""Helper function to get example L and LY data for testing."""

import numpy as np

# pylint: disable=invalid-name


def get_example_L_LY_data(
    len_psinorm: int, len_times: int, fill_value: float = 1.0
):
  """Returns example L and LY data for testing."""
  LY = {
      'rBt': np.full(len_times, fill_value).squeeze(),
      'aminor': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'rgeom': np.full((len_psinorm, len_times), 2.0 * fill_value).squeeze(),
      'TQ': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'FB': np.full(len_times, fill_value).squeeze(),
      'FA': np.full(len_times, fill_value).squeeze(),
      'Q0Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'Q1Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'Q2Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'Q3Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'Q4Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'Q5Q': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'ItQ': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'deltau': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'deltal': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'kappa': np.full((len_psinorm, len_times), fill_value).squeeze(),
      'epsilon': np.full((len_psinorm, len_times), fill_value).squeeze(),
      # When fill_value != 0 (i.e. intended to generate a standard geometry),
      # needs to be linspace to avoid drho_norm = 0.
      'FtPQ': (
          np.array([
              np.linspace(0, fill_value, len_psinorm) for _ in range(len_times)
          ]).T.squeeze()
      ),
      'zA': np.zeros(len_times).squeeze(),
      't': np.zeros(len_times).squeeze(),
      'lX': np.zeros(len_times, dtype=int).squeeze(),
  }
  L = {'pQ': np.linspace(0, 1, len_psinorm)}
  return L, LY
