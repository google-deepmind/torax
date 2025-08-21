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

"""Identical to test_iterhybrid_predictor_corrector but with TGLFNNukaea
transport"""
import copy
from pathlib import Path

from torax.tests.test_data import test_iterhybrid_predictor_corrector

base_path = Path(
    '/home/theo/documents/ukaea/tglfnn-ukaea/MultiMachineHyper_1Aug25'
)

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG['transport'] = {
    'model_name': 'tglfnn-ukaea',
    'machine': 'multimachine',
    'config_path': base_path / 'config.yaml',
    'stats_path': base_path / 'stats.json',
    'efe_gb_pt': base_path / 'regressor_efe_gb.pt',
    'efi_gb_pt': base_path / 'regressor_efi_gb.pt',
    'pfi_gb_pt': base_path / 'regressor_pfi_gb.pt',
    # Patches
    'apply_inner_patch': True,
    'D_e_inner': 0.25,
    'V_e_inner': 0.0,
    'chi_i_inner': 1.0,
    'chi_e_inner': 1.0,
    'rho_inner': 0.2,
    'apply_outer_patch': True,
    'D_e_outer': 0.1,
    'V_e_outer': 0.0,
    'chi_i_outer': 2.0,
    'chi_e_outer': 2.0,
    'rho_outer': 0.9,
    # Clipping
    'chi_min': 0.05,
    'chi_max': 100,
    'D_e_min': 0.05,
}

# Disable all but electron heat for the time being
CONFIG['numerics']['evolve_electron_heat'] = True
CONFIG['numerics']['evolve_ion_heat'] = False
CONFIG['numerics']['evolve_current'] = False
CONFIG['numerics']['evolve_density'] = False
