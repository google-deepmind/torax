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
    - calculate_P_LH: Calculates the H-mode transition power according to a
      given scaling law.
    - calculate_scaling_law_confinement_time: Calculates the predicted
      thermal energy confinement time from a given empirical scaling law.
"""

import dataclasses
import enum
import jax
from jax import numpy as jnp
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


class PLHScalingLaw(enum.StrEnum):
  MARTIN = 'martin'
  DELABIE = 'delabie'


class DivertorConfiguration(enum.StrEnum):
  HT = 'HT'
  VT = 'VT'


_P_LH_SCALING_PARAMS = {
    PLHScalingLaw.MARTIN: {
        # Equation 2 in Y.R. Martin and T. Takizuka, "Power requirement for
        # accessing the H-mode in ITER." Journal of Physics: Conference Series.
        # Vol. 123. No. 1. (2008)
        'prefactor': 0.0488,
        'Bt_exponent': 0.803,
        'ne_exponent': 0.717,
        'Meff_exponent': 1.0,
        'S_exponent': 0.941,
    },
    PLHScalingLaw.DELABIE: {
        # Equation 5 in E. Delabie et al. "Empirical scaling of the L–H
        # threshold power for metal wall tokamaks using a multi-device
        # database." Nuclear Fusion 66 (2026) 036016
        'prefactor': 0.0441,
        'Bt_exponent': 0.580,
        'ne_exponent': 1.08,
        'Meff_exponent': 0.975,
        'S_exponent': 1.0,
    },
}


@dataclasses.dataclass
class PLHAuxiliaryData:
  """Auxiliary data for P_LH calculations."""

  P_LH_high_density: jax.Array
  P_LH_low_density: jax.Array
  P_LH_min: jax.Array
  line_average_n_e_at_P_LH_min: jax.Array


def _calculate_P_LH_high_density(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    line_average_n_e: jax.Array,
    scaling_law: PLHScalingLaw,
    divertor_factor: float = 1.0,
    custom_prefactor: float = 1.0,
) -> jax.Array:
  """Calculates the H-mode transition power for the high density branch.

  See Eq. 3 in E. Delabie et al 2026 Nucl. Fusion 66 036016 for the general
  form. This unifies the Martin and Delabie scaling laws.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    line_average_n_e: Line average electron density in m^-3.
    scaling_law: Scaling law to use.
    divertor_factor: Factor D to account for divertor configuration.
    custom_prefactor: Prefactor to multiply the P_LH by.

  Returns:
    P_LH in W.
  """
  params = _P_LH_SCALING_PARAMS[scaling_law]

  line_avg_n_e_20 = line_average_n_e / 1e20  # [10^20 m^-3]
  Bt = geo.B_0  # [T]
  S = geo.g0_face[-1]  # Surface (not cross-section) area of LCFS [m^2]
  Meff = core_profiles.A_i  # Effective main ion atomic mass [amu]

  P_LH_MW = (
      custom_prefactor  # Prefactor from the user
      * params['prefactor']  # Prefactor from the scaling law
      * Bt ** params['Bt_exponent']
      * line_avg_n_e_20 ** params['ne_exponent']
      * (2.0 / Meff) ** params['Meff_exponent']
      * divertor_factor
      * S ** params['S_exponent']
  )

  return P_LH_MW * 1e6


def _calculate_line_average_n_e_at_P_LH_min(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Calculates the density at P_LH_min from equation 3 in Ryter 2014."""
  Ip_total = core_profiles.Ip_profile_face[..., -1]
  return (
      0.7
      * (Ip_total / 1e6) ** 0.34
      * geo.a_minor**-0.95
      * geo.B_0**0.62
      * (geo.R_major / geo.a_minor) ** 0.4
      * 1e19
  )


def calculate_P_LH(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    scaling_law: PLHScalingLaw,
    divertor_configuration: DivertorConfiguration = DivertorConfiguration.VT,
    prefactor: float = 1.0,
) -> tuple[jax.Array, PLHAuxiliaryData]:
  """Calculates the H-mode transition power from a given scaling law.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    scaling_law: Scaling law to use.
    divertor_configuration: Divertor configuration to use.
    prefactor: Prefactor to multiply the P_LH by.

  Returns:
    P_LH in W, and auxiliary data.
  """
  if scaling_law == PLHScalingLaw.DELABIE:
    if divertor_configuration == DivertorConfiguration.HT:
      divertor_factor = 1.0
    elif divertor_configuration == DivertorConfiguration.VT:
      divertor_factor = 1.93
    else:
      raise ValueError(
          'Unknown divertor configuration for Delabie scaling law:'
          f' {divertor_configuration}'
      )
  else:
    divertor_factor = 1.0

  line_average_n_e = math_utils.line_average(core_profiles.n_e.value, geo)
  line_average_n_e_at_P_LH_min = _calculate_line_average_n_e_at_P_LH_min(
      geo, core_profiles
  )

  # P_LH_high_density is the value given by the high density scaling law.
  P_LH_high_density = _calculate_P_LH_high_density(
      geo=geo,
      core_profiles=core_profiles,
      line_average_n_e=line_average_n_e,
      scaling_law=scaling_law,
      divertor_factor=divertor_factor,
      custom_prefactor=prefactor,
  )
  # P_LH_min is the value given by the high density scaling law evaluated at
  # line_average_n_e_at_P_LH_min.
  P_LH_min = _calculate_P_LH_high_density(
      geo=geo,
      core_profiles=core_profiles,
      line_average_n_e=line_average_n_e_at_P_LH_min,
      scaling_law=scaling_law,
      divertor_factor=divertor_factor,
      custom_prefactor=prefactor,
  )
  # Assume P_LH scales as n_e^-2 for the low density branch.
  P_LH_low_density = (
      P_LH_min * (line_average_n_e_at_P_LH_min / line_average_n_e) ** 2.0
  )

  P_LH = jnp.where(
      line_average_n_e > line_average_n_e_at_P_LH_min,
      P_LH_high_density,
      P_LH_low_density,
  )

  return P_LH, PLHAuxiliaryData(
      P_LH_high_density=P_LH_high_density,
      P_LH_low_density=P_LH_low_density,
      P_LH_min=P_LH_min,
      line_average_n_e_at_P_LH_min=line_average_n_e_at_P_LH_min,
  )


def calculate_scaling_law_confinement_time(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    P_loss: jax.Array,
    scaling_law: str,
) -> jax.Array:
  """Calculates the thermal energy confinement time for a given scaling law.

  Args:
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    P_loss: Plasma power loss in W.
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

  # Floor P_loss to a negligible but positive value. This is to avoid NaNs in
  # scaling law outputs.
  # Negative P_loss can arise during strong transients, when dW/dt > P_heat.
  # Scaling laws were not intended to be used in transient regimes, but we still
  # need to avoid NaNs.
  P_loss = jnp.maximum(P_loss, 1.0)

  scaled_Ip = core_profiles.Ip_profile_face[-1] / 1e6  # convert to MA
  scaled_Ploss = P_loss / 1e6  # convert to MW
  B = geo.B_0
  line_avg_n_e = (  # convert to 10^19 m^-3
      math_utils.line_average(core_profiles.n_e.value, geo) / 1e19
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
