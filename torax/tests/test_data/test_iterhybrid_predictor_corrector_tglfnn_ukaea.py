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
import os

from torax.tests.test_data import test_iterhybrid_predictor_corrector

# Note: This simulation uses a D-T plasma, but TGLFNNukaea is currently
# only designed for D only
CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

# Set to use TGLFNN
CONFIG['transport'] = {
    'model_name': 'tglfnn-ukaea',
    'machine': 'multimachine',
    # Inner patch
    'apply_inner_patch': True,
    'D_e_inner': 0.25,
    'V_e_inner': 0.0,
    'chi_i_inner': 1.0,
    'chi_e_inner': 1.0,
    'rho_inner': 0.2,  # radius below which patch transport is applied
    # Outer patch
    'apply_outer_patch': True,
    'D_e_outer': 0.1,
    'V_e_outer': 0.0,
    'chi_i_outer': 2.0,
    'chi_e_outer': 2.0,
    'rho_outer': 0.9,
    # Smoothing
    'smoothing_width': 0.1,
    'smooth_everywhere': False,
    # Clipping
    'chi_min': 0.05,
    'chi_max': 100,
    'D_e_min': 0.05,
    # Effective D, V method
    'An_min': 0.05,
    'DV_effective': True,
}
