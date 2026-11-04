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
"""Pydantic config for geometry."""

# pylint: disable=invalid-name

from typing import Annotated, TypeAlias

import pydantic
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry
from torax._src.geometry import eqdsk
from torax._src.geometry import imas
from torax._src.geometry import meq
from torax._src.geometry import tokamaker

GeometryConfig: TypeAlias = Annotated[
    circular_geometry.CircularConfig
    | chease.CheaseConfig
    | meq.MEQConfig
    | eqdsk.EQDSKConfig
    | imas.IMASConfig
    | tokamaker.TokaMakerConfig,
    pydantic.Field(discriminator='model_name'),
]
