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
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax.torax_pydantic import model_base
from torax.transport_model import pydantic_model as transport_model_pydantic_model


class ToraxConfig(model_base.BaseModelFrozen):
  """Base config class for Torax.

  Attributes:
    geometry: Config for the geometry.
    pedestal: Config for the pedestal model.
    stepper: Config for the stepper.
    time_step_calculator: Config for the time step calculator.
  """

  geometry: geometry_pydantic_model.Geometry
  pedestal: pedestal_pydantic_model.Pedestal
  stepper: stepper_pydantic_model.Stepper
  time_step_calculator: time_step_calculator_pydantic_model.TimeStepCalculator
  transport: transport_model_pydantic_model.Transport

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these nodes will be re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """
    self._update_fields(x)
