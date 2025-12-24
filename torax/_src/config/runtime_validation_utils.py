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

"""Utilities for validating the config inputs."""

from collections.abc import Mapping
import functools
import logging
from typing import Annotated, Any, Final, TypeAlias

import numpy as np
import pydantic
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic

_TOLERANCE: Final[float] = 1e-6

def _check_q_profile(config: Any) -> None:  # Replace Any with ToraxConfig when possible
    """Checks if the q-profile is unphysically low and raises a warning."""
    try:
        if hasattr(config, "profile_conditions") and hasattr(config, "geometry"):
            ip = getattr(config.profile_conditions.Ip, "value", None)
            bt = getattr(config.geometry.geometry_configs.config, "B_0", None)
            a = getattr(config.geometry.geometry_configs.config, "a_minor", None)
            r0 = getattr(config.geometry.geometry_configs.config, "R_major", None)
            if hasattr(config.geometry, "R_major") and hasattr(config.geometry, "a_minor"):
                r0 = getattr(config.geometry, "R_major", r0)
                a = getattr(config.geometry, "a_minor", a)
        else:
            gp = config.get("global_parameters", {}) if isinstance(config, dict) else {}
            geo = config.get("geometry", {}) if isinstance(config, dict) else {}
            ip = gp.get("plasma_current", {}).get("value") if isinstance(gp, dict) else None
            bt = gp.get("toroidal_field", {}).get("value") if isinstance(gp, dict) else None
            a = geo.get("a_minor", geo.get("minor_radius")) if isinstance(geo, dict) else None
            r0 = geo.get("R_major", geo.get("major_radius")) if isinstance(geo, dict) else None

        if ip is None or bt is None or a is None or r0 is None:
            return

        if hasattr(ip, "__len__") and getattr(ip, "shape", ()):
            ip_val = float(ip[0])
        else:
            ip_val = float(ip)
        bt_val = float(bt)
        a_val = float(a)
        r0_val = float(r0)

        q_estimate = (bt_val * a_val**2) / (ip_val * r0_val)

        if q_estimate < 1.0:
            logging.warning(
                f"Q-profile estimate is very low (q ~ {q_estimate:.2f}). "
                "This might indicate an unphysical configuration with too high "
                "plasma current and too low toroidal field."
            )
    except Exception:
        return

def _check_source_densities(config: Any) -> None:  # Replace Any with ToraxConfig when possible
    sources = None
    if isinstance(config, dict):
        sources = config.get("sources")
    else:
        sources = getattr(config, "sources", None)
    if sources is None:
        return
    try:
        if isinstance(sources, dict):
            power = sources.get("power_source_density", {}).get("value")
        else:
            power = getattr(getattr(sources, "power_source_density", None), "value", None)
        if power is not None and float(power) > 1e8:
            logging.warning("Power source density is very high (value=%s).", power)
    except Exception:
        pass
    try:
        if isinstance(sources, dict):
            particle = sources.get("particle_source_density", {}).get("value")
        else:
            particle = getattr(getattr(sources, "particle_source_density", None), "value", None)
        if particle is not None and float(particle) > 1e20:
            logging.warning(
                "Particle source density is very high (value=%s).", particle
            )
    except Exception:
        pass

def time_varying_array_defined_at_1(
    time_varying_array: torax_pydantic.TimeVaryingArray,
) -> torax_pydantic.TimeVaryingArray:
  """Validates the input for the TimeVaryingArray."""
  if not time_varying_array.right_boundary_conditions_defined:
    logging.debug("""Not defined at rho=1.0.""")
  return time_varying_array


def time_varying_array_bounded(
    time_varying_array: torax_pydantic.TimeVaryingArray,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
) -> torax_pydantic.TimeVaryingArray:
  """Validates the input for the TimeVaryingArray."""
  for t, (_, values) in time_varying_array.value.items():
    if not np.all(values >= lower_bound):
      raise ValueError(
          f'Some values are smaller than lower bound {lower_bound} at time'
          f' {t}: {values}'
      )
    if not np.all(values <= upper_bound):
      raise ValueError(
          f'Some values are larger than upper bound {upper_bound} at time'
          f' {t}: {values}'
      )
  return time_varying_array


TimeVaryingArrayDefinedAtRightBoundaryAndBounded: TypeAlias = Annotated[
    torax_pydantic.TimeVaryingArray,
    pydantic.AfterValidator(time_varying_array_defined_at_1),
    pydantic.AfterValidator(
        functools.partial(
            time_varying_array_bounded,
            lower_bound=1.0,
        )
    ),
]


def _ion_mixture_before_validator(value: Any) -> Any:
  """Validates the input for the IonMixtureType."""
  if isinstance(value, str):
    return {value: 1.0}
  return value


def _ion_mixture_after_validator(
    value: Mapping[str, torax_pydantic.TimeVaryingScalar],
) -> Mapping[str, torax_pydantic.TimeVaryingScalar]:
  """Validates the input for the IonMixtureType."""
  if not value:
    raise ValueError('The species dictionary cannot be empty.')

  # Check if all species keys are in the allowed list.
  invalid_ion_symbols = set(value.keys()) - constants.ION_SYMBOLS
  if invalid_ion_symbols:
    raise ValueError(
        f'Invalid ion symbols: {invalid_ion_symbols}. Allowed symbols are:'
        f' {constants.ION_SYMBOLS}'
    )

  time_arrays = [v.time for v in value.values()]
  fraction_arrays = [v.value for v in value.values()]

  # Check if all time arrays are equal
  if not all(np.array_equal(time_arrays[0], x) for x in time_arrays[1:]):
    raise ValueError(
        'All time indices for ion mixture fractions must be equal.'
    )

  # Check if the ion fractions sum to 1 at all times
  fraction_sum = np.sum(fraction_arrays, axis=0)
  if not np.allclose(fraction_sum, 1.0, rtol=_TOLERANCE):
    raise ValueError(
        'Fractional concentrations in an IonMixture must sum to 1 at all times.'
    )
  return value


IonMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.TimeVaryingScalar],
    pydantic.BeforeValidator(_ion_mixture_before_validator),
    pydantic.AfterValidator(_ion_mixture_after_validator),
]
