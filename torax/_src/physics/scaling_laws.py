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

"""Calculations related to empirical scaling laws.

Functions:
    - calculate_plh_scaling_factor: Calculates the H-mode transition power
      according to Martin 2008, and the density corresponding to the P_LH_min
      according to Ryter 2014.
    - calculate_scaling_law_confinement_time: Calculates the predicted
      thermal energy confinement time from a given empirical scaling law.
"""

import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def calculate_plh_scaling_factor(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Calculates the H-mode transition power scalings.

  See Y.R. Martin and Tomonori Takizuka.
  "Power requirement for accessing the H-mode in ITER."
  Journal of Physics: Conference Series. Vol. 123. No. 1. IOP Publishing, 2008.

  Only valid for hydrogenic isotopes and mixtures (H, D, T).
  Includes a simple inverse scaling of the factor to average isotope mass.

  For an overview see U Plank, U., et al. "Overview of L-to H-mode transition
  experiments at ASDEX Upgrade."
  Plasma Physics and Controlled Fusion 65.1 (2022): 014001.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.

  Returns:
    Tuple of: P_LH scaling factor for high density branch, minimum P_LH,
      P_LH = max(P_LH_min, P_LH_hi_dens) for practical use, and the density
      corresponding to the P_LH_min.
  """

  line_avg_n_e = math_utils.line_average(core_profiles.n_e.value, geo)

  # LH transition power for deuterium, in W. Eq 3 from Martin 2008.
  P_LH_hi_dens_D = (
      2.15
      * (line_avg_n_e / 1e20) ** 0.782
      * geo.B_0**0.772
      * geo.a_minor**0.975
      * geo.R_major**0.999
      * 1e6
  )

  # Scale to average isotope mass.
  A_deuterium = constants.ION_PROPERTIES_DICT['D']['A']
  P_LH_hi_dens = P_LH_hi_dens_D * A_deuterium / core_profiles.A_i

  Ip_total = core_profiles.Ip_profile_face[..., -1]

  # Calculate density corresponding to P_LH_min from Eq 3 Ryter 2014
  n_e_min_P_LH = (
      0.7
      * (Ip_total / 1e6) ** 0.34
      * geo.a_minor**-0.95
      * geo.B_0**0.62
      * (geo.R_major / geo.a_minor) ** 0.4
      * 1e19
  )
  # Calculate P_LH_min at n_e_min from Eq 4 Ryter 2014
  P_LH_min_D = (
      0.36
      * (Ip_total / 1e6) ** 0.27
      * geo.B_0**1.25
      * geo.R_major**1.23
      * (geo.R_major / geo.a_minor) ** 0.08
      * 1e6
  )
  P_LH_min = P_LH_min_D * A_deuterium / core_profiles.A_i
  P_LH = jnp.maximum(P_LH_min, P_LH_hi_dens)
  return P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH


def calculate_scaling_law_confinement_time(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    Ploss: jax.Array,
    scaling_law: str,
) -> jax.Array:
  """Calculates the thermal energy confinement time for a given empirical scaling law.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    Ploss: Plasma power loss in W.
    scaling_law: Scaling law to use.

  Returns:
    Thermal energy confinement time in s.
  """
  scaling_params = {
      'H89P': {
          # From Yushmanov et al, Nuclear Fusion, vol. 30, no. 10, pp. 4-6, 1990
          'prefactor': 0.038128,
          'Ip_exponent': 0.85,
          'B_exponent': 0.2,
          'line_avg_n_e_exponent': 0.1,
          'Ploss_exponent': -0.5,
          'R_exponent': 1.5,
          'inverse_aspect_ratio_exponent': 0.3,
          'elongation_exponent': 0.5,
          'effective_mass_exponent': 0.50,
          'triangularity_exponent': 0.0,
      },
      'H98': {
          # H98 empirical confinement scaling law:
          # ITER Physics Expert Groups on Confinement and Transport and
          # Confinement Modelling and Database, Nucl. Fusion 39 2175, 1999
          # Doyle et al, Nucl. Fusion 47 (2007) S18–S127, Eq 30
          'prefactor': 0.0562,
          'Ip_exponent': 0.93,
          'B_exponent': 0.15,
          'line_avg_n_e_exponent': 0.41,
          'Ploss_exponent': -0.69,
          'R_exponent': 1.97,
          'inverse_aspect_ratio_exponent': 0.58,
          'elongation_exponent': 0.78,
          'effective_mass_exponent': 0.19,
          'triangularity_exponent': 0.0,
      },
      'H97L': {
          # From the ITER L-mode confinement database.
          # S.M. Kaye et al 1997 Nucl. Fusion 37 1303, Eq 7
          'prefactor': 0.023,
          'Ip_exponent': 0.96,
          'B_exponent': 0.03,
          'line_avg_n_e_exponent': 0.4,
          'Ploss_exponent': -0.73,
          'R_exponent': 1.83,
          'inverse_aspect_ratio_exponent': -0.06,
          'elongation_exponent': 0.64,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.0,
      },
      'H20': {
          # Updated ITER H-mode confinement database, using full dataset.
          # G. Verdoolaege et al 2021 Nucl. Fusion 61 076006, Eq 7
          'prefactor': 0.053,
          'Ip_exponent': 0.98,
          'B_exponent': 0.22,
          'line_avg_n_e_exponent': 0.24,
          'Ploss_exponent': -0.669,
          'R_exponent': 1.71,
          'inverse_aspect_ratio_exponent': 0.35,
          'elongation_exponent': 0.80,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.36,  # (1+delta)^exponent
      },
  }

  if scaling_law not in scaling_params:
    raise ValueError(f'Unknown scaling law: {scaling_law}')

  params = scaling_params[scaling_law]

  scaled_Ip = core_profiles.Ip_profile_face[-1] / 1e6  # convert to MA
  scaled_Ploss = Ploss / 1e6  # convert to MW
  B = geo.B_0
  line_avg_n_e = (  # convert to 10^19 m^-3
      math_utils.line_average(core_profiles.n_e.value, geo)
      / 1e19
  )
  R = geo.R_major
  inverse_aspect_ratio = geo.a_minor / geo.R_major

  # Effective elongation definition. This is a different definition than
  # the standard definition used in geo.elongation.
  elongation = geo.area_face[-1] / (jnp.pi * geo.a_minor**2)
  # TODO(b/317360834): extend when multiple ions are supported.
  effective_mass = core_profiles.A_i
  triangularity = geo.delta_face[-1]

  tau_scaling = (
      params['prefactor']
      * scaled_Ip ** params['Ip_exponent']
      * B ** params['B_exponent']
      * line_avg_n_e ** params['line_avg_n_e_exponent']
      * scaled_Ploss ** params['Ploss_exponent']
      * R ** params['R_exponent']
      * inverse_aspect_ratio ** params['inverse_aspect_ratio_exponent']
      * elongation ** params['elongation_exponent']
      * effective_mass ** params['effective_mass_exponent']
      * (1 + triangularity) ** params['triangularity_exponent']
  )
  return tau_scaling
