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
"""Base class for QLKNN Models."""

import abc
from typing import TypeAlias

import jax
from torax._src.transport_model import qualikiz_based_transport_model

ModelOutput: TypeAlias = dict[str, jax.Array]
InputsAndRanges: TypeAlias = dict[str, dict[str, float]]


class BaseQLKNNModel(abc.ABC):
  """Base class for QLKNN Models."""

  def __init__(self, path: str, name: str):
    self.path = path
    self.name = name

  @property
  @abc.abstractmethod
  def inputs_and_ranges(self) -> InputsAndRanges:
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(
      self,
      inputs: jax.Array,
  ) -> ModelOutput:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_model_inputs_from_qualikiz_inputs(
      self, qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs
  ) -> jax.Array:
    raise NotImplementedError()
