# Copyright 2026 DeepMind Technologies Limited
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
"""Combined model for atomic cooling rates."""

import enum
import functools
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src.physics.radiation import mavrin_coronal_cooling_rate
from torax._src.physics.radiation import mavrin_noncoronal_cooling_rate

# Coronal equilibrium is reached when n_e * tau_p ~ 10^19 m^-3 s.
# Above this limit, we are in the coronal regime, below, we are in the
# non-coronal regime.
# A. Mavrin. "Radiative Cooling Rates for Low-Z Impurities in Non-coronal
# Equilibrium State", J Fusion Energ (2017) 36:161–172
# DOI 10.1007/s10894-017-0136-z
# TODO(b/434175938): (v2) Rename to n_e_tau for consistency.

_NE_TAU_CORONAL_LIMIT = 1e19

# pylint: disable=invalid-name


class MavrinModelType(enum.StrEnum):
  CORONAL = 'coronal'
  NONCORONAL = 'noncoronal'


@functools.partial(
    jax.jit,
    static_argnames=[
        'ion_symbol',
    ],
)
def calculate_cooling_rate(
    T_e: array_typing.FloatVector,
    ion_symbol: str,
    # TODO(b/434175938): (v2) Rename to n_e_tau for consistency.
    ne_tau: array_typing.FloatScalar = _NE_TAU_CORONAL_LIMIT,
) -> array_typing.FloatVector:
  """Compute the cooling rate for a single impurity species.

  Different models are used in different temperature regimes:
  - For T_e > T_min_coronal we use the coronal Mavrin model
  - For T_min_noncoronal < T_e < T_min_coronal, we use the noncoronal Mavrin
    model
  - For T_e < T_min_noncoronal, we use 0

  The coronal Mavrin model fit is valid for temperatures between 100eV - 100keV,
  and the noncoronal Mavrin model fit is valid for temperatures between
  ~1eV-~10keV (with some variation between different elements). To maximise the
  predictive range, we combine the two models by using a smooth transition to
  blend their predictions.

  Args:
    T_e: Electron temperature [keV]. Can be any sized array, e.g. on cell grid,
      face grid, or a single scalar.
    ion_symbol: Ion symbol of the impurity species.
    ne_tau: Effective residence parameter [m^-3 s]. Only used for the noncoronal
      model.

  Returns:
    Cooling rate [W m^-3].
  """

  if ion_symbol not in constants.ION_SYMBOLS:
    raise ValueError(
        f'Invalid ion symbol: {ion_symbol}. Allowed symbols are :'
        f' {constants.ION_SYMBOLS}'
    )

  # Alias He3 and He4 to He as they are chemically identical
  if ion_symbol in ('He3', 'He4'):
    ion_symbol_lookup = 'He'
  else:
    ion_symbol_lookup = ion_symbol

  # Cooling rate in the coronal regime
  T_e_min_coronal = mavrin_coronal_cooling_rate.MIN_TEMPERATURES[
      ion_symbol_lookup
  ]
  coronal_cooling_rate = calculate_mavrin_cooling_rate(
      T_e, ion_symbol_lookup, MavrinModelType.CORONAL
  )

  # Cooling rate in the noncoronal regime
  if ion_symbol_lookup not in mavrin_noncoronal_cooling_rate.COEFFS.keys():
    # For ions that do not have a noncoronal model, we linearly extrapolate
    # from (T_e_min_coronal, LZ_min_coronal) to (0, 0).
    noncoronal_cooling_rate = (
        T_e
        / T_e_min_coronal
        * calculate_mavrin_cooling_rate(
            T_e_min_coronal, ion_symbol_lookup, MavrinModelType.CORONAL
        )
    )
  else:
    noncoronal_cooling_rate_from_mavrin_fit = calculate_mavrin_cooling_rate(
        T_e, ion_symbol_lookup, MavrinModelType.NONCORONAL, ne_tau
    )

    # For temperatures below the Mavrin noncoronal fit range, we extrapolate the
    # cooling rate linearly from the minimum.
    T_e_min_noncoronal = mavrin_noncoronal_cooling_rate.MIN_TEMPERATURES[
        ion_symbol_lookup
    ]
    min_noncoronal_cooling_rate = calculate_mavrin_cooling_rate(
        T_e_min_noncoronal,
        ion_symbol_lookup,
        MavrinModelType.NONCORONAL,
        ne_tau,
    )
    extrapolated_noncoronal_cooling_rate = min_noncoronal_cooling_rate * (
        T_e / T_e_min_noncoronal
    )
    noncoronal_cooling_rate = jnp.where(
        T_e >= T_e_min_noncoronal,
        noncoronal_cooling_rate_from_mavrin_fit,
        extrapolated_noncoronal_cooling_rate,
    )

  # Smoothly transition between the noncoronal and coronal models based on the
  # minimum temperature of the coronal model.
  return math_utils.smoothstep_transition(
      T_e,
      smoothing_start=T_e_min_coronal / 1.5,
      smoothing_end=T_e_min_coronal * 1.5,
      y_left=noncoronal_cooling_rate,
      y_right=coronal_cooling_rate,
      log_scale=True,
  )


def calculate_mavrin_cooling_rate(
    T_e: array_typing.FloatVector,
    ion_symbol: str,
    model_type: MavrinModelType,
    ne_tau: array_typing.FloatScalar = _NE_TAU_CORONAL_LIMIT,
) -> array_typing.FloatVector:
  """Compute the cooling rate for a single impurity species using Mavrin fits."""
  # Alias He3 and He4 to He as they are chemically identical
  if ion_symbol in ('He3', 'He4'):
    ion_symbol = 'He'

  match model_type:
    case MavrinModelType.CORONAL:
      cooling_rate_module = mavrin_coronal_cooling_rate
    case MavrinModelType.NONCORONAL:
      cooling_rate_module = mavrin_noncoronal_cooling_rate
    case _:
      allowed_models = ', '.join([v.name for v in MavrinModelType])
      raise ValueError(
          f'Invalid model type: {model_type}. Allowed models are:'
          f' {allowed_models}'
      )

  if (
      model_type is MavrinModelType.NONCORONAL
      and ion_symbol not in cooling_rate_module.COEFFS.keys()
  ):
    # If the ion is not supported by the edge radiation model, we assume it
    # negligibly contributes to the edge physics (radiation or Z_eff/dilution in
    # the divertor). This is a good assumption for heavy impurities like W,
    # which this case covers. This behaviour is silent to avoid log spam.
    return jnp.zeros_like(T_e)

  # Restrict temperature to region of validity for Mavrin fits
  T_e_min = cooling_rate_module.MIN_TEMPERATURES[ion_symbol]
  T_e_max = cooling_rate_module.MAX_TEMPERATURES[ion_symbol]
  T_e = jnp.clip(T_e, T_e_min, T_e_max)

  # Restrict residence parameter to below the coronal limit
  ne_tau = jnp.clip(ne_tau, a_max=_NE_TAU_CORONAL_LIMIT)

  # Gather coefficients for each temperature
  # If the ion has different coefficients for different temperature ranges, we
  # need to find which interval each T_e value falls into. Otherwise, we can
  # just use the first set.
  if ion_symbol in cooling_rate_module.TEMPERATURE_INTERVALS:
    intervals = cooling_rate_module.TEMPERATURE_INTERVALS[ion_symbol]
    interval_indices = jnp.searchsorted(intervals, T_e)
  else:
    interval_indices = jnp.zeros_like(T_e, dtype=jnp.int32)

  # Select the appropriate coefficients based on the interval indices
  coeffs_in_range = jnp.take(
      cooling_rate_module.COEFFS[ion_symbol], interval_indices, axis=1
  )

  return cooling_rate_module.evaluate_polynomial_fit(
      T_e, ne_tau / _NE_TAU_CORONAL_LIMIT, coeffs_in_range
  )
