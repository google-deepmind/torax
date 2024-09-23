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

"""Functions for adding post-processed outputs to the simulation state."""

# In torax/post_processing.py

import dataclasses
import jax
from torax import array_typing
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import state

_trapz = jax.scipy.integrate.trapezoid


@jax_utils.jit
def _compute_pressure(
    core_profiles: state.CoreProfiles,
) -> tuple[array_typing.ArrayFloat, ...]:
  """Calculates pressure from density and temperatures.

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.

  Returns:
    p_el: Electron pressure [Pa]
    p_ion: Ion pressure [Pa]
    p_tot: Total pressure [Pa]
  """
  ne = core_profiles.ne.face_value()
  ni = core_profiles.ni.face_value()
  nimp = core_profiles.nimp.face_value()
  temp_ion = core_profiles.temp_ion.face_value()
  temp_el = core_profiles.temp_el.face_value()
  p_el = ne * temp_el * constants.CONSTANTS.keV2J * core_profiles.nref
  p_ion = (
      (ni + nimp) * temp_ion * constants.CONSTANTS.keV2J * core_profiles.nref
  )
  p_tot = p_el + p_ion
  return p_el, p_ion, p_tot


@jax_utils.jit
def _compute_stored_thermal_energy(
    p_el: array_typing.ArrayFloat,
    p_ion: array_typing.ArrayFloat,
    p_tot: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> tuple[array_typing.ScalarFloat, ...]:
  """Calculates stored thermal energy from pressures.

  Args:
    p_el: Electron pressure [Pa]
    p_ion: Ion pressure [Pa]
    p_tot: Total pressure [Pa]
    geo: Geometry object

  Returns:
    wth_el: Electron thermal stored energy [J]
    wth_ion: Ion thermal stored energy [J]
    wth_tot: Total thermal stored energy [J]
  """
  wth_el = _trapz(1.5 * p_el * geo.vpr_face, geo.rho_face_norm)
  wth_ion = _trapz(1.5 * p_ion * geo.vpr_face, geo.rho_face_norm)
  wth_tot = _trapz(1.5 * p_tot * geo.vpr_face, geo.rho_face_norm)

  return wth_el, wth_ion, wth_tot


def make_outputs(
    sim_state: state.ToraxSimState, geo: geometry.Geometry
) -> state.ToraxSimState:
  """Calculates post-processed outputs based on the latest state.

  Called at the beginning and end of each `sim.run_simulation` step.
  Args:
    sim_state: The state to add outputs to.
    geo: Geometry object

  Returns:
    sim_state: A ToraxSimState object, with any updated attributes.
  """
  p_el, p_ion, p_tot = _compute_pressure(sim_state.core_profiles)
  wth_el, wth_ion, wth_tot = _compute_stored_thermal_energy(
      p_el, p_ion, p_tot, geo
  )

  return dataclasses.replace(
      sim_state,
      post_processed_outputs=state.PostProcessedOutputs(
          pressure_thermal_ion_face=p_ion,
          pressure_thermal_el_face=p_el,
          pressure_thermal_tot_face=p_tot,
          wth_thermal_ion=wth_ion,
          wth_thermal_el=wth_el,
          wth_thermal_tot=wth_tot,
      ),
  )
