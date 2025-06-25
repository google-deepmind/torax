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

"""Script for plotting the combined transport model in the docs."""
from typing import Sequence

from absl import app
import matplotlib.pyplot as plt
import torax
from torax._src.torax_pydantic import model_config

_FILE_PATH = '/tmp/combined_transport_example.png'


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = {
      'profile_conditions': {},
      'numerics': {},
      'plasma_composition': {},
      'geometry': {
          'geometry_type': 'circular',
          'n_rho': 30,  # for higher resolution plotting
      },
      'pedestal': {
          'model_name': 'set_T_ped_n_ped',
          'set_pedestal': True,
          'rho_norm_ped_top': 0.9,
          'n_e_ped': 0.8,
          'n_e_ped_is_fGW': True,
      },
      'neoclassical': {},
      'sources': {},
      'solver': {},
      'transport': {
          'model_name': 'combined',
          'transport_models': [
              {
                  'model_name': 'constant',
                  'chi_i': 1.0,
                  'rho_max': 0.3,
              },
              {
                  'model_name': 'constant',
                  'chi_i': 2.0,
                  'rho_min': 0.2,
              },
          ],
          'pedestal_transport_models': [
              {
                  'model_name': 'constant',
                  'chi_i': 0.5,
              },
          ],
      },
  }
  torax_config = model_config.ToraxConfig.from_dict(config)
  data_tree, _ = torax.run_simulation(torax_config)
  plt.figure(figsize=(8, 2))
  plt.plot(
      data_tree.rho_face_norm,
      data_tree.profiles.chi_turb_i.sel(time=2, method='nearest'),
  )
  plt.xlabel('rho_norm_face')
  plt.ylabel(r'chi_turb_i [$m^2/s$]')
  plt.xlim(0, 1)
  plt.ylim(0, None)
  plt.title('Combined chi_turb_i profile')
  plt.tight_layout()
  plt.savefig(_FILE_PATH)


if __name__ == '__main__':
  app.run(main)
