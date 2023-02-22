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

"""Compare implicit vs explicit coupling terms.

Run the simulation with high qei_coeff to induce instability.
Check whether this is alleviated by switching to implicit coupling.
"""

from typing import Sequence

from absl import app
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import torax
from torax import state as state_lib
from torax.stepper import linear_theta_method


def _run_sim(
    coupling_use_explicit_source: bool, qei_mult: float
) -> tuple[torax.State, torax.Geometry, jnp.ndarray]:
  """Run the simulation.

  Args:
    coupling_use_explicit_source: See config.Config.
    qei_mult: See config.Config.

  Returns:
    states: One State per time step.
    geo: Geometry created by the simulation.
    t: History of time coordinates for each State.
  """
  unstable_pars = {
      'set_pedestal': False,
      'fGW': 1.0,
      'Qei_mult': qei_mult,
      't_final': 3.0,
      'solver': {
          'use_pereverzev': False,
          'coupling_use_explicit_source': coupling_use_explicit_source,
      },
  }

  config = torax.Config()
  config = torax.recursive_replace(config, **unstable_pars)
  geo = torax.build_circular_geometry(config)

  sim = torax.build_sim_from_config(
      config, geo, linear_theta_method.LinearThetaMethod
  )
  torax_outputs = sim.run()
  state_history = state_lib.build_state_history_from_outputs(torax_outputs)
  t = state_lib.build_time_history_from_outputs(torax_outputs)
  return state_history, geo, t


def main(_: Sequence[str]) -> None:
  qei_mult = 2.0

  e_states, e_geo, e_t = _run_sim(True, qei_mult)
  e_states_1, _, _ = _run_sim(True, 1.0)
  i_states, _, i_t = _run_sim(False, qei_mult)
  assert isinstance(e_t, jnp.ndarray), type(e_t)
  assert np.allclose(e_t, i_t)

  plot_cell_over_time = False
  if plot_cell_over_time:
    for r_idx in range(e_geo.mesh.nx):

      def get(states, r_idx):
        nt = e_t.size
        return [states[i].temp_ion.value[r_idx] for i in range(nt)]

      ion_temp_explicit = get(e_states, r_idx)
      ion_temp_explicit_1 = get(e_states_1, r_idx)
      ion_temp_implicit = get(i_states, r_idx)
      plt.xlabel('t')
      plt.ylabel('temp_ion')
      plt.title(f'Cell {r_idx}, Qei_mult={qei_mult}')
      plt.plot(e_t, ion_temp_explicit, label='Explicit')
      plt.plot(e_t, ion_temp_explicit_1, label='Explicit (Qei_mult=1)')
      plt.plot(e_t, ion_temp_implicit, label='Implicit')
      plt.legend()
      plt.show()

  plt.title(f'Qei_mult={qei_mult}, t={e_t[-1]}')
  plt.xlabel('Cell')
  plt.ylabel('T')
  plt.plot(e_states.temp_el.value[-1], label='temp_el explicit')
  plt.plot(e_states.temp_ion.value[-1], label='temp_ion explicit')
  plt.plot(i_states.temp_el.value[-1], label='temp_el implicit')
  plt.plot(i_states.temp_ion.value[-1], label='temp_ion implicit')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  app.run(main)
