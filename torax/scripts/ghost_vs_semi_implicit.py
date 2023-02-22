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

"""Compare ghost cell vs semi-implicit convection boundary condition modes.

This experiment runs a test6_no_pedestal to introduce instability
when using the semi-implicit scheme as used by FiPy.
This instability is fixed by changing to ghost cell mode.
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
    mode: str,
) -> tuple[torax.State, torax.Geometry, jnp.ndarray]:
  """Run the simulation.

  Args:
    mode: 'ghost' or 'semi-implicit'. Passed through to `fvm.convection_terms`
      for both `dirichlet_mode` and `neumann_mode`.

  Returns:
    states: One State per time step.
    geo: Geometry created by the simulation.
    t: History of time coordinates for each State.
  """
  unstable_pars = {
      'transport_model': 'CGM',
      'set_pedestal': False,
      'fGW': 1.0,
      't_final': 1.0,
      'solver': {
          'coupling_use_explicit_source': True,
          'convection_dirichlet_mode': mode,
          'convection_neumann_mode': mode,
      },
      'bootstrap_mult': 0.0,
  }

  config = torax.Config()
  config = torax.recursive_replace(config, **unstable_pars)
  geo = torax.build_circular_geometry(config)

  sim = torax.sim.build_sim_from_config(
      config,
      geo,
      linear_theta_method.LinearThetaMethod,
  )
  torax_outputs = sim.run()
  state_history = state_lib.build_state_history_from_outputs(torax_outputs)
  t = state_lib.build_time_history_from_outputs(torax_outputs)
  return state_history, geo, t


def main(_: Sequence[str]) -> None:
  g_states, g_geo, g_t = _run_sim('ghost')
  f_states, _, f_t = _run_sim('semi-implicit')
  if not np.allclose(g_t[-1], f_t[-1]):
    print(
        'Warning: the final time steps are not perfectly comparable.\n'
        f'Ghost cell mode terminated at t={g_t[-1]}.\n'
        f'Semi-implicit mode (emulating FiPy) terminated at t={f_t[-1]}.\n'
    )

  plot_cell_over_time = False
  if plot_cell_over_time:
    for r_idx in range(g_geo.mesh.nx):
      ion_temp_ghost = g_states.temp_ion[:, r_idx]
      ion_temp_fipy = f_states.temp_ion[:, r_idx]
      plt.xlabel('t')
      plt.ylabel('temp_ion')
      plt.title(f'Cell {r_idx}')
      plt.plot(g_t, ion_temp_ghost, label='Ghost')
      plt.plot(f_t, ion_temp_fipy, label='FiPy')
      plt.legend()
      plt.show()

  plt.title(f't={g_t[-1]}')
  plt.subplot(2, 1, 1)
  plt.xlabel('Cell')
  plt.ylabel('T')
  plt.plot(g_states.temp_el.value[-1], label='temp_el ghost')
  plt.plot(g_states.temp_ion.value[-1], label='temp_ion ghost')
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.xlabel('Cell')
  plt.ylabel('T')
  plt.plot(f_states.temp_el.value[-1], label='temp_el FiPy')
  plt.plot(f_states.temp_ion.value[-1], label='temp_ion FiPy')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  app.run(main)
