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

"""Pydantic config for MHD models."""

import chex
import pydantic
from torax._src.mhd import base
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.mhd.sawtooth import pydantic_model as sawtooth_pydantic_model
from torax._src.torax_pydantic import torax_pydantic


class MHD(torax_pydantic.BaseModelFrozen):
  """Config for MHD models.

  Attributes:
    sawtooth: Config for sawtooth models.
  """

  sawtooth: sawtooth_pydantic_model.SawtoothConfig | None = pydantic.Field(
      default=None
  )

  def build_mhd_models(self) -> base.MHDModels:
    """Builds and returns a container with instantiated MHD model objects."""
    sawtooth_model = (
        None if self.sawtooth is None else self.sawtooth.build_models()
    )
    return base.MHDModels(sawtooth_models=sawtooth_model)

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> mhd_runtime_params.DynamicMHDParams:
    """Builds and returns a container with dynamic runtime params for MHD models."""

    return mhd_runtime_params.DynamicMHDParams(**{
        mhd_model_name: mhd_model_config.build_dynamic_params(t)
        for mhd_model_name, mhd_model_config in self.__dict__.items()
        if mhd_model_config is not None
    })
