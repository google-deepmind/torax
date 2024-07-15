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
"""Class for handling QLKNN10D models."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import Any, Callable

import flax.linen as nn
import immutabledict
import jax
import numpy as np
from torax.transport_model import base_qlknn_model


# Move this to common lib.
_ACTIVATION_FNS: Mapping[str, Callable[[jax.Array], jax.Array]] = (
    immutabledict.immutabledict({
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
        'none': lambda x: x,
    })
)


class MLP(nn.Module):
  hidden_sizes: list[int]
  activations: list[str]

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    for act, size in zip(self.activations, self.hidden_sizes):
      x = _ACTIVATION_FNS[act](nn.Dense(size)(x))
    return x


class QuaLiKizNDNN:
  """Jax implementation for QLKNN10D inference."""

  def __init__(
      self,
      model_config: dict[str, Any],
  ):
    self._model_config = model_config

    self._feature_names = model_config.get('feature_names')
    self._target_names = model_config.get('target_names')

    self._feature_prescale_factor = self._load_prescale(
        'prescale_factor', self._feature_names
    )
    self._feature_prescale_bias = self._load_prescale(
        'prescale_bias', self._feature_names
    )
    self._target_prescale_factor = self._load_prescale(
        'prescale_factor', self._target_names
    )
    self._target_prescale_bias = self._load_prescale(
        'prescale_bias', self._target_names
    )

    activations = model_config['hidden_activation'] + [
        model_config['output_activation']
    ]
    hidden_sizes = []
    params = {}
    for i in range(len(activations)):
      weights = np.array(model_config[f'layer{i+1}/weights/Variable:0'])
      params[f'Dense_{i}'] = {
          'bias': np.array(model_config[f'layer{i+1}/biases/Variable:0']),
          'kernel': weights,
      }
      hidden_sizes.append(weights.shape[1])
      del model_config[f'layer{i+1}/biases/Variable:0']
      del model_config[f'layer{i+1}/weights/Variable:0']
    self._params = {'params': params}
    self._model = MLP(hidden_sizes=hidden_sizes, activations=activations)

  def _load_prescale(self, key: str, names: list[str]) -> np.ndarray:
    return np.array([self._model_config[key][k] for k in names])[
        np.newaxis, :
    ]

  def __call__(
      self,
      inputs: jax.Array,
  ) -> jax.Array:
    """Calculate the outputs given specific inputs."""

    inputs = (
        self._feature_prescale_factor * inputs + self._feature_prescale_bias
    )
    outputs = self._model.apply(self._params, inputs)
    outputs = (
        outputs - self._target_prescale_bias
    ) / self._target_prescale_factor

    return outputs

  @classmethod
  def from_json(cls, json_file) -> QuaLiKizNDNN:
    with open(json_file) as file_:
      model_dict = json.load(file_)
    return cls(model_dict)


class QLKNN10D(base_qlknn_model.BaseQLKNNModel):
  """Class holding QLKNN10D networks.

  Attributes:
    model_path: Path to qlknn-hyper
    net_itgleading: ITG Qi net
    net_itgqediv: ITG Qe/Qi net
    net_temleading: TEM Qe net
    net_temqediv: TEM Qi/Qe net
    net_etgleading: ETG Qe net
    net_temqidiv: TEM Qi/Qe net
    net_tempfediv: Tem pfe/Qe net
    net_etgleading: ITG Qe/Qi net
    net_itgpfediv: ITG pfe/Qi net
  """

  def __init__(self, model_path: str):
    super().__init__(version='10D')
    self.model_path = model_path
    self.net_itgleading = self._load('efiitg_gb.json')
    self.net_itgqediv = self._load('efeitg_gb_div_efiitg_gb.json')
    self.net_temleading = self._load('efetem_gb.json')
    self.net_temqidiv = self._load('efitem_gb_div_efetem_gb.json')
    self.net_tempfediv = self._load('pfetem_gb_div_efetem_gb.json')
    self.net_etgleading = self._load('efeetg_gb.json')
    self.net_itgpfediv = self._load('pfeitg_gb_div_efiitg_gb.json')

  def _load(self, path) -> QuaLiKizNDNN:
    full_path = os.path.join(self.model_path, path)
    return QuaLiKizNDNN.from_json(full_path)

  def predict(
      self,
      inputs: jax.Array,
  ) -> base_qlknn_model.ModelOutput:
    """Feed forward through the network and compute fluxes."""

    model_output = {}
    model_output['qi_itg'] = self.net_itgleading(inputs).clip(0)
    model_output['qe_itg'] = (
        self.net_itgqediv(inputs) * model_output['qi_itg']
    )
    model_output['pfe_itg'] = (
        self.net_itgpfediv(inputs) * model_output['qi_itg']
    )
    model_output['qe_tem'] = self.net_temleading(inputs).clip(0)
    model_output['qi_tem'] = (
        self.net_temqidiv(inputs) * model_output['qe_tem']
    )
    model_output['pfe_tem'] = (
        self.net_tempfediv(inputs) * model_output['qe_tem']
    )
    model_output['qe_etg'] = self.net_etgleading(inputs).clip(0)
    return model_output
