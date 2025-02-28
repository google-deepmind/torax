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

"""Pydantic config for Torax."""

from typing import Any, Mapping
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_model_config
from torax.time_step_calculator import config as time_step_calculator_config
from torax.torax_pydantic import model_base


class ToraxConfig(model_base.BaseModelFrozen):
  """Base config class for Torax.

  Attributes:
    time_step_calculator: Config for the time step calculator.
    pedestal: Config for the pedestal model.
  """

  geometry: geometry_pydantic_model.Geometry
  pedestal: pedestal_model_config.PedestalModel
  time_step_calculator: time_step_calculator_config.TimeStepCalculator

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these nodes will be re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        value is the new value for that field.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """
    self._update_fields(x)
