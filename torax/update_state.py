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

"""Function to update the state."""

import dataclasses
from torax import config_slice
from torax import physics
from torax import state
from torax.fvm import cell_variable


def update_core_profiles(
    core_profiles: state.CoreProfiles,
    x_new: tuple[cell_variable.CellVariable, ...],
    evolving_names: tuple[str, ...],
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Args:
    core_profiles: The old set of core plasma profiles.
    x_new: The new values of the evolving variables.
    evolving_names: The names of the evolving variables.
    dynamic_config_slice: The dynamic config slice.
  """

  def get_update(x_new, var):
    """Returns the new value of `var`."""
    if var in evolving_names:
      return x_new[evolving_names.index(var)]
    # `var` is not evolving, so its new value is just its old value
    return getattr(core_profiles, var)

  temp_ion = get_update(x_new, 'temp_ion')
  temp_el = get_update(x_new, 'temp_el')
  psi = get_update(x_new, 'psi')
  ne = get_update(x_new, 'ne')
  ni = dataclasses.replace(
      core_profiles.ni,
      value=ne.value
      * physics.get_main_ion_dilution_factor(
          dynamic_config_slice.plasma_composition.Zimp,
          dynamic_config_slice.plasma_composition.Zeff,
      ),
  )

  return dataclasses.replace(
      core_profiles,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
  )
