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

"""Demonstration of a grid search for constrained optimization.

The goal is to maximize fusion power at `t_final` wrt `fext`, `rext`, and
`Ip`, while respecting constraints on q.

Specifically, we ensure that q_cell > 1 for r / Rmin > 0.1

Args:
  path: The filepath to save the output of the grid search.  The results can be
    plotted using `grid_search_plot.py`.
"""

import dataclasses
import functools
from typing import Sequence

from absl import app
import jax
from jax import numpy as jnp
import torax
from torax import state as state_lib
from torax.sources import fusion_heat_source
from torax.stepper import linear_theta_method

# pylint:disable=invalid-name
# Names like "Ip" are chosen for consistency with standard physics notation


def fusion_power(
    fext: float,
    rext: float,
    Ip: float,
    base_config: torax.Config,
    geo: torax.Geometry,
):
  """Run the simulation and calculate fusion power for the given scenario.

  Args:
    fext: Override `base_config.fext` with this value
    rext: Override `base_config.rext` with this value
    Ip: Override `base_config.Ip` with this value
    base_config: Starting Config
    geo: Magnetic geometry.

  Returns:
    fusion_power: fusion power at the last time step.
    state_history: A State in history mode, logging the whole simulation.
  """

  config = dataclasses.replace(
      base_config,
      fext=fext,
      rext=rext,
      Ip=Ip,
  )

  sim = torax.build_sim_from_config(
      config,
      geo,
      linear_theta_method.LinearThetaMethod,
  )
  torax_outputs = sim.run()
  state_history = state_lib.build_state_history_from_outputs(torax_outputs)

  # while loop terminated at t_final or slightly after, just
  # use the last state in the array
  last_state = state_history.index(-1)
  fusion = fusion_heat_source.calc_fusion(geo, last_state, config.nref)

  return fusion, state_history


def main(argv: Sequence[str]) -> None:
  _, path = argv

  # For this demo we use the configuration from sim test9, but include the
  # bootstrap current
  override = {
      "transport_model": "qlknn",
      "Ti_bound_left": 8.0,
      "Te_bound_left": 8.0,
      "current_eq": True,
      # To shorten current diffusion time
      "resistivity_mult": 100,
      # set flat Ohmic current to provide larger range of current evolution
      "nu": 0.0,
      "t_final": 2.0,
      "solver": {
          "coupling_use_explicit_source": False,
          # As of 2023-01-31, JAX does not support differentiation through
          # tridiagonal_solve
          "use_tridiagonal_solve": False,
      },
  }
  base_config = torax.Config()
  base_config = torax.recursive_replace(base_config, **override)

  geo = torax.build_circular_geometry(base_config)
  fp_while = functools.partial(fusion_power, base_config=base_config, geo=geo)
  # The while loop implementation calls jit on the main loop
  # internally, no need to do so here.

  # Find the index of the start of the q constraint region
  start_r_idx = 0
  while geo.r_face[start_r_idx] < 0.1:
    start_r_idx += 1

  best_fp = None
  with open(path, "w") as file:
    file.write("fext\trext\tIp\tfusion_power\tmin q(r>0.1)\n")
    for fext in jnp.linspace(0.0, 0.7, 6):
      for rext in jnp.linspace(0.0, 1.75, 6):
        for Ip in jnp.linspace(8.0, 15.0, 6):
          try:
            fp, states = fp_while(fext, rext, Ip)
          except jax.lib.xla_extension.XlaRuntimeError as e:
            # Some points in the grid search may get NaNs etc., we just want
            # to record they were infeasible and move on.
            print("Ignoring exception:", e)
            file.write(f"{fext}\t{rext}\t{Ip}\tNone\tNone\tNone\n")
            print(fext, rext, Ip, "invalid")
            continue
          min_qr = states.q_face[:, start_r_idx:].min()
          if min_qr >= 1.0:
            if best_fp is None or fp > best_fp:
              best_fp = fp
          file.write(f"{fext}\t{rext}\t{Ip}\t{fp}\t{min_qr}\n")
          print(f"{fext}\t{rext}\t{Ip}\t{fp}\t{min_qr}\n")
  print("best_fp: ", best_fp)


if __name__ == "__main__":
  app.run(main)
