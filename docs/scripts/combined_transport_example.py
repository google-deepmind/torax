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

"""Script for plotting the combined transport model in the docs"""

import matplotlib.pyplot as plt

import torax
from torax._src.torax_pydantic import model_config
from torax.examples import basic_config
import copy

config = copy.deepcopy(basic_config.CONFIG)
config['geometry']['n_rho'] = 30  # increased for plotting
config['numerics']['t_final'] = 2
config['transport'] = {
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
            'rho_max': 0.5,
        },
        {
            'model_name': 'constant',
            'chi_i': 0.5,
            'rho_min': 0.5,
            'rho_max': 1.0,
        },
    ],
}
torax_config = model_config.ToraxConfig.from_dict(config)
data_tree, state_history = torax.run_simulation(torax_config)
plt.figure(figsize=(12, 3))
plt.plot(
    data_tree.rho_face_norm,
    data_tree.profiles.chi_turb_i.sel(time=2, method='nearest'),
)
plt.xlabel('rho_norm_face')
plt.ylabel('chi_i')
plt.xlim(0, 1)
plt.ylim(0, None)
plt.title('Combined chi_i profile (t=2)')
plt.tight_layout()
plt.savefig('docs/images/combined_transport_example.png')
