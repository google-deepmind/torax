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

"""Tests BgB transport.

- BgB transport model
- Linear stepper with Pereverzev-Corrigan
- Chi time step calculator
- Circular geometry
- No bootstrap, default ion-el heat source, default qei source, default jext
"""

CONFIG = {
    "runtime_params": {
        "numerics": {
            "fixed_dt": 1e-2,
            "t_final": 2,
            "ion_heat_eq": True,
            "el_heat_eq": True,
            "dens_eq": True,
            "current_eq": True,
        },
    },
    "geometry": {
        "geometry_type": "circular",
    },
    "sources": {
        "j_bootstrap": {},
        "generic_ion_el_heat_source": {},
        "qei_source": {},
        "jext": {},
    },
    "transport": {
        "transport_model": "bohm-gyrobohm",
        "chimin": 0.05,
        "bohm-gyrobohm_params": {
            "chi_e_bohm_coeff": 1.0,
            "chi_e_gyrobohm_coeff": 1.0,
            "chi_i_bohm_coeff": 1.0,
            "chi_i_gyrobohm_coeff": 1.0,
            "d_face_c1": 0.0,
            "d_face_c2": 0.0,
        },
    },
    "stepper": {
        "stepper_type": "linear",
        "predictor_corrector": False,
        "use_pereverzev": True,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}
