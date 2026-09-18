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
"""Zeros model for neoclassical transport."""
from typing import Annotated, Literal

from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import base
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs
from typing_extensions import override


class ZerosModel(base.NeoclassicalTransportModel):
  """Zeros model for neoclassical transport."""

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_coeffs.NeoclassicalTransport:
    """Calculates neoclassical transport."""
    return transport_coeffs.NeoclassicalTransport.zeros(geometry)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class ZerosModelConfig(base.NeoclassicalTransportModelConfig):
  """Config for the Zeros model implementation of neoclassical transport."""

  model_name: Annotated[Literal['zeros'], torax_pydantic.JAX_STATIC] = 'zeros'

  def build_model(self) -> ZerosModel:
    return ZerosModel()
