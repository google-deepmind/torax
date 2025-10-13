# Copyright 2025 DeepMind Technologies Limited
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
"""Register a transport model with TORAX."""

from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model_base


def register_transport_model(
    pydantic_model_class: type[pydantic_model_base.TransportBase],
):
  """Registers a transport model with TORAX."""
  model_config.ToraxConfig.model_fields['transport'].annotation |= (
      pydantic_model_class
  )
  model_config.ToraxConfig.model_rebuild(force=True)
