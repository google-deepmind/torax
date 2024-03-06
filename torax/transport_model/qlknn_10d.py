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

import os

import jax.numpy as jnp
from qlknn.models import ffnn
from torax.transport_model import _qlknn_np
from torax.transport_model import base_qlknn_model


class _WrappedQLKNN:
  """A wrapper around a QLKNN10D network, to be called with JAX tracers.

  Attributes:
    network: The raw `qlknn` network.
  """

  def __init__(self, network: ffnn.QuaLiKizNDNN):
    self.network = network

  def __call__(self, x: jnp.ndarray):
    """Call the network, with a JAX argument."""
    return self.network.get_output(x, safe=False, output_pandas=False)


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

  def _load(self, path):
    full_path = os.path.join(self.model_path, path)
    raw = ffnn.QuaLiKizNDNN.from_json(full_path, np=_qlknn_np)
    return _WrappedQLKNN(raw)

  def predict(
      self,
      inputs: jnp.ndarray,
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
