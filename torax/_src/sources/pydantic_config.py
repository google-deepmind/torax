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

"""Pydantic config for source models."""

from collections.abc import Mapping
from typing import Any, Self

import immutabledict
import pydantic
from torax._src.sources import base
from torax._src.sources import bremsstrahlung_heat_sink as bremsstrahlung_heat_sink_lib
from torax._src.sources import cyclotron_radiation_heat_sink as cyclotron_radiation_heat_sink_lib
from torax._src.sources import electron_cyclotron_source as electron_cyclotron_source_lib
from torax._src.sources import fusion_heat_source as fusion_heat_source_lib
from torax._src.sources import gas_puff_source as gas_puff_source_lib
from torax._src.sources import generic_current_source as generic_current_source_lib
from torax._src.sources import generic_ion_el_heat_source as generic_ion_el_heat_source_lib
from torax._src.sources import generic_particle_source as generic_particle_source_lib
from torax._src.sources import ohmic_heat_source as ohmic_heat_source_lib
from torax._src.sources import pellet_source as pellet_source_lib
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import runtime_params
from torax._src.sources import source_models
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax._src.sources.ion_cyclotron_source import scaled_profile
from torax._src.sources.ion_cyclotron_source import toric_nn
from torax._src.torax_pydantic import torax_pydantic


def _source_field(
    default_config_if_enabled: type[base.SourceConfigBase],
) -> Any:
  """Creates a Pydantic field for an optional source config.

  Args:
    default_config_if_enabled: The source config to use if the field is
      provided as a dict without a `model_name`.

  Returns:
    A Pydantic field with appropriate discriminator and default values set in
    the schema extra.
  """
  return pydantic.Field(
      discriminator='model_name',
      default=None,
      json_schema_extra={
          'default_model_name': (
              default_config_if_enabled.model_fields['model_name'].default
          )
      },
  )


class Sources(torax_pydantic.BaseModelFrozen):
  """Config for source models.

  Each standard source field defaults to `None` (disabled) when omitted. When a
  source is configured with a mapping that omits `model_name`, the default model
  specified via `_source_field` for that source is used.
  """

  ei_exchange: qei_source_lib.QeiSourceConfig = torax_pydantic.ValidatedDefault(
      {'mode': 'ZERO'}
  )
  # keep-sorted start
  bremsstrahlung: (
      bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig | None
  ) = _source_field(
      default_config_if_enabled=bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig
  )
  cyclotron_radiation: (
      cyclotron_radiation_heat_sink_lib.CyclotronRadiationHeatSinkConfig | None
  ) = _source_field(
      default_config_if_enabled=cyclotron_radiation_heat_sink_lib.CyclotronRadiationHeatSinkConfig
  )
  ecrh: electron_cyclotron_source_lib.ElectronCyclotronSourceConfig | None = (
      _source_field(
          default_config_if_enabled=electron_cyclotron_source_lib.ElectronCyclotronSourceConfig
      )
  )
  fusion: fusion_heat_source_lib.FusionHeatSourceConfig | None = _source_field(
      default_config_if_enabled=fusion_heat_source_lib.FusionHeatSourceConfig
  )
  gas_puff: gas_puff_source_lib.GasPuffSourceConfig | None = _source_field(
      default_config_if_enabled=gas_puff_source_lib.GasPuffSourceConfig
  )
  generic_current: (
      generic_current_source_lib.GenericCurrentSourceConfig | None
  ) = _source_field(
      default_config_if_enabled=generic_current_source_lib.GenericCurrentSourceConfig
  )
  generic_heat: (
      generic_ion_el_heat_source_lib.GenericIonElHeatSourceConfig | None
  ) = _source_field(
      default_config_if_enabled=generic_ion_el_heat_source_lib.GenericIonElHeatSourceConfig
  )
  generic_particle: (
      generic_particle_source_lib.GenericParticleSourceConfig | None
  ) = _source_field(
      default_config_if_enabled=generic_particle_source_lib.GenericParticleSourceConfig
  )
  icrh: (
      toric_nn.ToricNNIonCyclotronSourceConfig
      | scaled_profile.ScaledProfileIonCyclotronSourceConfig
      | None
  ) = _source_field(
      default_config_if_enabled=toric_nn.ToricNNIonCyclotronSourceConfig
  )
  impurity_radiation: (
      impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig
      | impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig
      | None
  ) = _source_field(
      default_config_if_enabled=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig
  )
  ohmic: ohmic_heat_source_lib.OhmicHeatSourceConfig | None = _source_field(
      default_config_if_enabled=ohmic_heat_source_lib.OhmicHeatSourceConfig
  )
  pellet: pellet_source_lib.PelletSourceConfig | None = _source_field(
      default_config_if_enabled=pellet_source_lib.PelletSourceConfig
  )
  # keep-sorted end

  @pydantic.model_validator(mode='before')
  @classmethod
  def _set_default_model_names(cls, x: dict[str, Any]) -> dict[str, Any]:
    """Populates default `model_name`s for source mappings that omit them."""
    constructor_data = dict(x)
    for k, v in x.items():
      if isinstance(v, Mapping) and 'model_name' not in v:
        field = cls.model_fields.get(k)
        if field is not None and isinstance(field.json_schema_extra, Mapping):
          constructor_data[k] = {
              'model_name': field.json_schema_extra['default_model_name'],
              **v,
          }
    return constructor_data

  @pydantic.model_validator(mode='after')
  def _set_exclude_impurity_bremsstrahlung(self) -> Self:
    """Auto-configure bremsstrahlung when Mavrin impurity radiation is active.

    When the Mavrin model is active, it already accounts for impurity
    bremsstrahlung via higher-fidelity ADAS data. To avoid double-counting,
    we set exclude_impurity_bremsstrahlung=True on the bremsstrahlung source
    so it only computes main-ion bremsstrahlung.

    Returns:
      Self for method chaining.
    """
    if isinstance(
        self.bremsstrahlung,
        bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig,
    ) and isinstance(
        self.impurity_radiation,
        impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
    ):
      bremsstrahlung_active = (
          self.bremsstrahlung.mode != runtime_params.Mode.ZERO
      )
      impurity_active = self.impurity_radiation.mode != runtime_params.Mode.ZERO

      if bremsstrahlung_active and impurity_active:
        object.__setattr__(
            self.bremsstrahlung,
            'exclude_impurity_bremsstrahlung',
            True,
        )

    return self

  def build_models(self) -> source_models.SourceModels:
    """Builds and returns a container with instantiated source model objects."""
    standard_sources = {}
    for k, v in dict(self).items():
      # ei_exchange is handled separately above.
      if k == 'ei_exchange':
        continue
      else:
        if v is not None:
          source = v.build_source()
          if k in standard_sources:
            raise ValueError(
                f'Trying to add another source with the same name: {k}.'
            )
          standard_sources[k] = source
    qei_source_model = self.ei_exchange.build_source()
    # Qei is a special source that is not in standard_sources.
    # It has its own attribute in SourceModels.
    return source_models.SourceModels(
        qei_source=qei_source_model,
        standard_sources=immutabledict.immutabledict(standard_sources),
    )

  @property
  def source_model_config(self) -> dict[str, base.SourceConfigBase]:
    return {
        k: v
        for k, v in self.__dict__.items()
        if isinstance(v, base.SourceConfigBase)
    }
