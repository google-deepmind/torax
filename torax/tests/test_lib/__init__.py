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

"""A library with test functionality shared across tests."""

# Mapping function from sim test names to sim reference file name.
# By default the test name and file names match. However, several
# cases are nonstandard. The following dict defines the nonstandard
# mappings.
_REF_MAP_OVERRIDES = {
    'test_crank_nicolson': 'test_implicit.nc',
    'test_arraytimestepcalculator': 'test_qei.nc',
    'test_absolute_generic_current_source': 'test_psi_and_heat.nc',
    'test_newton_raphson_zeroiter': 'test_psi_and_heat.nc',
}


def get_data_file(test_name: str) -> str:
  return _REF_MAP_OVERRIDES.get(test_name, f'{test_name}.nc')
