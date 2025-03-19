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

"""Generic ion/electron heat source."""

from __future__ import annotations


"""Generic heat source for both ion and electron heat."""

import dataclasses
from typing import Any, Callable, Literal, Optional, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import array_typing
from torax import jax_utils
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.config import base as config_base
from torax.geometry import geometry
from torax.sources import formulas
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.torax_pydantic import torax_pydantic

T = TypeVar('T')

# Constants matching the "H_fraction" for the 3 electron density regions in METIS.
# For H-mode, the electron density is ~0.65 Greenwald.
# For the Low density region (ne < 0.4 nGW), H_fraction=0.74
# For the ITER baseline density (0.4 nGW < ne < 0.8 nGW), H_fraction = 0.95
# For the high density region (0.8 nGW < ne < 1.0 nGW), H_fraction = 0.85


class GenericIonElHeatSourceConfig(base.SourceModelBase):
  """Configuration for the generic ion/electron heat source.

  Attributes:
    source_name: Name of the source
    rsource: center of gaussian profile
    w: width of gaussian profile
    Ptot: Total heating: high default based on total ITER power including alphas
    el_heat_fraction: Electron heating fraction
    absorption_fraction: Fraction of absorbed power
  """
  source_name: Literal['generic_ion_el_heat_source'] = (
      'generic_ion_el_heat_sink'
  )
  rsource: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.3
  )
  w: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.2)
  Ptot: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  el_heat_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  # TODO(b/817): Add appropriate pydantic validation for absorption_fraction
  absorption_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def build_source(self) -> source_lib.Source:
    """Builds a source object from the model config."""
    return GenericIonElHeatSource(model_func=model_func)

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    """Returns the model function for the source."""
    return model_func

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> runtime_params_lib.DynamicRuntimeParams:
    """Builds dynamic runtime parameters for the source."""
    return DynamicRuntimeParams(
        w=self.w.get_value(t),
        rsource=self.rsource.get_value(t),
        Ptot=self.Ptot.get_value(t),
        el_heat_fraction=self.el_heat_fraction.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
        prescribed_values=self.prescribed_values.get_value(t) if self.mode == runtime_params_lib.Mode.PRESCRIBED else None,
    )


@dataclasses.dataclass(frozen=True)

# pylint: disable=invalid-name
@chex.dataclass(frozen=True)

class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  w: array_typing.ScalarFloat = 0.2
  rsource: array_typing.ScalarFloat = 0.3
  Ptot: array_typing.ScalarFloat = 120e6
  el_heat_fraction: array_typing.ScalarFloat = 0.66666
  absorption_fraction: array_typing.ScalarFloat = 1.0
  prescribed_values: Optional[Any] = None

  def tree_flatten(self):
    """Returns a flattened version of the tree."""
    children = (self.w, self.rsource, self.Ptot, self.el_heat_fraction, 
               self.absorption_fraction, self.prescribed_values)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Creates a new instance of the class from flattened data."""
    return cls(
        w=children[0],
        rsource=children[1],
        Ptot=children[2],
        el_heat_fraction=children[3],
        absorption_fraction=children[4],
        prescribed_values=children[5],
    )

jax.tree_util.register_pytree_node_class(DynamicRuntimeParams)


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(config_base.RuntimeParametersConfig['RuntimeParamsProvider']):
  """Runtime parameters for the generic heat source."""

  # External heat source parameters
  # Gaussian width in normalized radial coordinate
  w: config_base.ScalarOrTimeTrace = 0.2
  # Gaussian center
  rsource: config_base.ScalarOrTimeTrace = 0.3
  # Total power injected
  Ptot: config_base.ScalarOrTimeTrace = 120e6
  # Heat deposition to electrons. Ion fraction = 1 - el_heat_fraction.
  el_heat_fraction: config_base.ScalarOrTimeTrace = 0.66666
  # Fraction of power that is absorbed in the core
  absorption_fraction: config_base.ScalarOrTimeTrace = 1.0

  def get_provider(self) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(
        w=self.w,
        rsource=self.rsource,
        Ptot=self.Ptot,
        el_heat_fraction=self.el_heat_fraction,
        absorption_fraction=self.absorption_fraction,
    )


@dataclasses.dataclass(kw_only=True)
class RuntimeParamsProvider(
    config_base.RuntimeParametersProvider[DynamicRuntimeParams]
):
  """Provider for generic heat source runtime parameters."""
  w: config_base.ScalarOrTimeTrace
  rsource: config_base.ScalarOrTimeTrace
  Ptot: config_base.ScalarOrTimeTrace
  el_heat_fraction: config_base.ScalarOrTimeTrace
  absorption_fraction: config_base.ScalarOrTimeTrace

  def __call__(self, t: float) -> DynamicRuntimeParams:
    """Returns the runtime parameters at time t."""
    return DynamicRuntimeParams(
        w=config_base.time_evaluate(self.w, t),
        rsource=config_base.time_evaluate(self.rsource, t),
        Ptot=config_base.time_evaluate(self.Ptot, t),
        el_heat_fraction=config_base.time_evaluate(self.el_heat_fraction, t),
        absorption_fraction=config_base.time_evaluate(
            self.absorption_fraction, t
        ),
    )


def calc_generic_heat_source(
    geo: geometry.Geometry,
    rsource: float,
    w: float,
    Ptot: float,
    el_heat_fraction: float,
    absorption_fraction: float,
) -> tuple[chex.Array, chex.Array]:
  """Computes ion/electron heat source terms.

  We model this as a Gaussian heat deposition in the core.

  Args:
    geo: magnetic geometry
    rsource: center of the Gaussian in normalized radial coord
    w: width of deposition profile
    Ptot: total heating
    el_heat_fraction: fraction of heating deposited on electrons
    absorption_fraction: fraction of absorbed power

  Returns:
    A 2-tuple of the deposited powers in ion and electron channels (ion first)
  """
  # Calculate heat profile.
  # Apply absorption_fraction to the total power
  absorbed_power = Ptot * absorption_fraction
  profile = formulas.gaussian_profile(geo, center=rsource, width=w, total=absorbed_power)
  pion = (1 - el_heat_fraction) * profile
  pel = el_heat_fraction * profile
  return (pion, pel)


def model_func_impl(
    static_runtime_params_slice,
    dynamic_runtime_params_slice,
    geo,
    source_name,
    core_profiles,
    calculated_source_profiles,
):
    """Generic heat source model function.

    Args:
        static_runtime_params_slice: Static runtime parameters.
        dynamic_runtime_params_slice: Dynamic runtime parameters slice.
        geo: Geometry of the torus.
        source_name: Name of the source.
        core_profiles: Core plasma profiles.
        calculated_source_profiles: Already calculated source profiles if they exist.

    Returns:
        Tuple of arrays containing heat source profiles for ion and electron channels.
    """
    runtime_params = dynamic_runtime_params_slice.sources[source_name]
    return calc_generic_heat_source(
        geo=geo,
        rsource=runtime_params.rsource,
        w=runtime_params.w,
        Ptot=runtime_params.Ptot,
        el_heat_fraction=runtime_params.el_heat_fraction,
        absorption_fraction=runtime_params.absorption_fraction,
    )

# Don't use jit at all due to complexity with geometry objects
model_func = model_func_impl


class GenericIonElHeatSource(source_lib.Source):
  """Generic heating source for electron and ion."."""

  SOURCE_NAME = 'generic_ion_el_heat_source'

  def __init__(self, model_func: Optional[source_lib.SourceProfileFunction] = None):
    """Initializes the source.

    Args:
      model_func: The function to use for computing the source profile.
    """
    super().__init__(model_func=model_func)

  @property
  def source_name(self) -> str:
    """Returns the name of the source."""
    return self.SOURCE_NAME

  def init(
      self, gen_params: runtime_params_lib.DynamicRuntimeParams | None
  ) -> None:
    """Gets model parameters."""
    self.rsource = gen_params.rsource if gen_params else 0.3  # type: ignore
    self.width = gen_params.w if gen_params else 0.2  # type: ignore
    self.Ptot = gen_params.Ptot if gen_params else 120e6  # type: ignore
    self.el_heat_fraction = (
        gen_params.el_heat_fraction if gen_params else 0.66666
    )  # type: ignore
    self.absorption_fraction = (
        gen_params.absorption_fraction if gen_params else 1.0
    )  # type: ignore

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    """Returns which core profiles this source affects."""
    return (
        source_lib.AffectedCoreProfile.TEMP_EL,
        source_lib.AffectedCoreProfile.TEMP_ION,
    )


# Add alias for backward compatibility
GenericIonElectronHeatSource = GenericIonElHeatSource

class GenericIonElHeatSourceConfig(base.SourceModelBase):
  """Configuration for the GenericIonElHeatSource.

  Attributes:
    w: Gaussian width in normalized radial coordinate
    rsource: Source Gaussian central location (in normalized r)
    Ptot: Total heating: high default based on total ITER power including alphas
    el_heat_fraction: Electron heating fraction
  """

  source_name: Literal['generic_ion_el_heat_source'] = (
      'generic_ion_el_heat_source'
  )
  w: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.25)
  rsource: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.0
  )
  Ptot: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      120e6
  )
  el_heat_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return default_formula

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        w=self.w.get_value(t),
        rsource=self.rsource.get_value(t),
        Ptot=self.Ptot.get_value(t),
        el_heat_fraction=self.el_heat_fraction.get_value(t),
    )

  def build_source(self) -> GenericIonElectronHeatSource:
    return GenericIonElectronHeatSource(model_func=self.model_func)

