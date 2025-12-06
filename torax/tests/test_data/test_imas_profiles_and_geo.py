# Copyright 2025 DeepMind Technologies Limited
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

"""Tests combined IMAS geometry and profiles.

Load existing test_iterhybrid_predictor_corrector config and updates it with
IMAS geometry and profiles. Different ways of dumping IMAS data into config are
used: via geometry loader, directly load all keys from the IMAS loader or just
a specific key.
"""
import copy

from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader
from torax.tests.test_data import test_iterhybrid_predictor_corrector

imas_profiles1 = loader.load_imas_data(
    "core_profiles_ddv4_iterhybrid_rampup_conditions.nc", "core_profiles"
)
imas_profiles2 = loader.load_imas_data(
    "core_profiles_15MA_DT_50_50_flat_top_slice.nc", "core_profiles"
)
imas_data1 = core_profiles.profile_conditions_from_IMAS(imas_profiles1)
imas_data2 = core_profiles.plasma_composition_from_IMAS(
    imas_profiles2, t_initial=0.0
)

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)
CONFIG["geometry"] = {
    "geometry_type": "imas",
    "imas_filepath": "ITERhybrid_COCOS17_IDS_ddv4.nc",
    "Ip_from_parameters": False,
}
# Dump all profile_conditions from the first IDS loaded.
CONFIG["profile_conditions"] = {
    **imas_data1,
    "initial_psi_mode": "geometry",
}
# Load just a specific key from the other IDS: impurity species dict.
CONFIG["plasma_composition"] = {
    "main_ion": {"D": 0.5, "T": 0.5},  # (bundled isotope average)
    "impurity": {
        "species": {
            **imas_data2["impurity"]["species"],
            "He3": None,
        },
        # Manually set one impurity ratio to None to agree with
        # the n_e_ratios_Z_eff impurity mode.
        "impurity_mode": "n_e_ratios_Z_eff",
    },
    "Z_eff": 1.6,  # sets impurity density
}
