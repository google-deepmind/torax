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

"""Same as test_iterhybrid_predictor_corrector but with Lin Liu ECRH model."""
import copy
from torax.tests.test_data import test_iterhybrid_predictor_corrector


CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

# Remove generic current source, to be replaced by ECRH.
CONFIG['sources'].pop('generic_current_source')

# Set generic ion el heat source to a pure ion source.
# Electron component is handled by the ECRH source.
# This way, the total ion and electron heating is unchanged,
# but the ECCD current drive is model-based.
CONFIG['sources']['generic_ion_el_heat_source']['el_heat_fraction'] = 0.0
CONFIG['sources']['generic_ion_el_heat_source']['Ptot'] = 16.32e6

# Set the ECRH source.
CONFIG['sources']['electron_cyclotron_source'] = {
    'gaussian_ec_power_density_width': 0.075,
    'gaussian_ec_power_density_location': 0.36,
    'gaussian_ec_total_power': 34.68e6,
}
